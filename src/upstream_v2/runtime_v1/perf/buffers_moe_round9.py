"""Isolated research: bulk pointwise, unchanged per-expert GEMMs and BF16 sums."""
import os
from pathlib import Path
import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from route_moe import route_indices
from fused_decode_round7 import decode

MODE = 'triton'
VARIANT = 1
_extension = None

def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        here=Path(__file__).resolve().parent
        build=here/'buffers-round9-build'; build.mkdir(exist_ok=True)
        os.environ['MAX_JOBS']='1'
        _extension=load(name='solos_buffers_round9',sources=[str(here/'expert_buffers_round9.cpp')],
            build_directory=str(build),extra_cflags=['-O2','-g0'],with_cuda=False,verbose=True)
    return _extension

@triton.jit
def _ordered_sum(values, indices, output, N:tl.constexpr, D:tl.constexpr,
                 K:tl.constexpr, REVERSE:tl.constexpr, BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=off//D; col=off%D; ok=token<N
    acc=tl.full((BLOCK,),0,tl.float32)
    for j in range(K):
        k=K-1-j if REVERSE else j
        row=tl.load(indices+token*K+k,mask=ok,other=0)
        val=tl.load(values+row*D+col,mask=ok,other=0).to(tl.float32)
        # Original index_add updates BF16 output after every expert, not just
        # once after reducing top-k. Preserve every intermediate rounding.
        acc=(acc+val).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output+off,acc,mask=ok)

def reduce_routes(values,indices,reverse):
    n,k=indices.shape; d=values.shape[-1]
    if MODE=='aten':
        out=values.new_zeros(n,d)
        for j in (range(k-1,-1,-1) if reverse else range(k)):
            out.add_(values.index_select(0,indices[:,j]))
        return out
    out=values.new_empty(n,d)
    _ordered_sum[(triton.cdiv(n*d,256),)](values,indices,out,n,d,k,reverse,256,
                                       enable_fp_fusion=False)
    return out

class Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights,positions,tokens,indices,counts,packed):
        ctx.counts,ctx.packed,ctx.variant=counts,packed,VARIANT
        decoded=decode(packed)
        values=extension().forward(x,weights,positions,tokens,counts,
            decoded['gate_proj'],decoded['up_proj'],decoded['down_proj'],ctx.variant)
        ctx.save_for_backward(weights,positions,tokens,indices,*values[1:])
        return reduce_routes(values[0],indices,False)

    @staticmethod
    def backward(ctx,upstream):
        weights,positions,tokens,indices,g,u,a,y=ctx.saved_tensors
        with torch.autocast('cuda',enabled=False):
            decoded=decode(ctx.packed)
            dx,gw=extension().backward(weights,positions,tokens,ctx.counts,
                decoded['gate_proj'],decoded['up_proj'],decoded['down_proj'],
                upstream,g,u,a,y,ctx.variant)
            gx=reduce_routes(dx,indices,True)
        return gx,gw,None,None,None,None,None

def forward(self,hidden_states):
    shape=hidden_states.shape; x=hidden_states.reshape(-1,shape[-1])
    logits=self.gate(x)
    weights=F.softmax(logits,dim=1,dtype=torch.float)
    weights,selected=torch.topk(weights,self.top_k,dim=-1)
    if self.norm_topk_prob:
        weights=weights/weights.sum(dim=-1,keepdim=True)
    counts,ks,tokens=route_indices(selected,self.num_experts)
    positions=tokens*self.top_k+ks
    inverse=torch.empty_like(positions)
    inverse.scatter_(0,positions,torch.arange(positions.numel(),device=positions.device))
    # Each row enumerates the routed entries in ascending expert order.
    indices=inverse.view_as(selected).gather(1,selected.argsort(dim=1))
    out=Experts.apply(x,weights.to(x.dtype),positions,tokens,indices,counts,self._packed_nf4)
    return out.view(shape),logits
