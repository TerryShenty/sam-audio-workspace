"""Measured research candidate: fused decode, ragged grouped GEMM, ordered sums.

NF4/BF16 and all trainable paths are retained. Grouped GEMM can change rounding;
the user explicitly allows measured numerical differences. Never self-installs.
"""
import torch
from torch.nn import functional as F
from fused_decode_round7 import decode
from pointwise_moe_round8 import reduce_routes

import nf4_grouped_round13b as backend
PAD = 16
FP32_OUTPUT = False

class Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights,selected,packed,pad,fp32):
        n,d=x.shape;k=selected.shape[1];e=packed['gate_proj'].experts
        flat=selected.T.reshape(-1);order=torch.argsort(flat,stable=True)
        expert=flat[order];counts=torch.bincount(flat,minlength=e)
        padded=(counts+pad-1)//pad*pad;ends=padded.cumsum(0).to(torch.int32)
        rows=int(ends[-1]);starts=counts.cumsum(0)-counts
        dst=ends.long()[expert]-padded[expert]+torch.arange(flat.numel(),device=x.device)-starts[expert]
        token=order.remainder(n);slot=torch.div(order,n,rounding_mode='floor')
        positions=token*k+slot
        inverse=torch.empty_like(positions);inverse.scatter_(0,positions,dst)
        indices=inverse.view_as(selected).gather(1,selected.argsort(dim=1))
        routed=x.new_zeros(rows+16,d);routed.index_copy_(0,dst,x.index_select(0,token))
        def mm(a,name):
            out=backend.grouped_mm(a.contiguous(),packed[name],ends,rows,True)
            out[rows:]=0
            return out
        g=mm(routed,'gate_proj');u=mm(routed,'up_proj');a=F.silu(g)
        y=mm(a*u,'down_proj')
        w=weights.new_zeros(rows+16,1)
        w.index_copy_(0,dst,weights.flatten().index_select(0,positions).unsqueeze(-1))
        ctx.save_for_backward(weights,positions,token,dst,indices,ends,g,u,a,y)
        ctx.packed,ctx.rows,ctx.fp32=packed,rows,fp32
        return reduce_routes(y*w,indices,False)

    @staticmethod
    def backward(ctx,upstream):
        weights,positions,token,dst,indices,ends,g,u,a,y=ctx.saved_tensors
        with torch.autocast('cuda',enabled=False):
            dy=upstream.index_select(0,token)
            dw=(dy*y.index_select(0,dst)).sum(-1)
            gw=torch.zeros_like(weights);gw.flatten().index_copy_(0,positions,dw)
            w=weights.flatten().index_select(0,positions).unsqueeze(-1)
            weighted=torch.zeros_like(y);weighted.index_copy_(0,dst,dy*w)
            def mm(v,name):
                out=backend.grouped_mm(v.contiguous(),ctx.packed[name],ends,ctx.rows,False)
                out[ctx.rows:]=0
                return out
            dm=mm(weighted,'down_proj')
            dg=torch.ops.aten.silu_backward.default(dm*u,g);du=dm*a
            dx=mm(dg,'gate_proj')+mm(du,'up_proj')
            gx=reduce_routes(dx,indices,True)
        return gx,gw,None,None,None,None

def forward(self,hidden_states):
    shape=hidden_states.shape;x=hidden_states.reshape(-1,shape[-1])
    logits=self.gate(x);weights=F.softmax(logits,dim=1,dtype=torch.float32)
    weights,selected=torch.topk(weights,self.top_k,dim=-1)
    if self.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
    out=Experts.apply(x,weights.to(x.dtype),selected,self._packed_nf4,PAD,FP32_OUTPUT)
    return out.view(shape),logits
