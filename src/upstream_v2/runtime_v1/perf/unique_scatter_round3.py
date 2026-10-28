"""Exact-shape ATen expert execution with lossless quantized saved weights.

The C++ loop uses the same linear/silu/multiply/index_add operations as the
original. Autograd remains PyTorch autograd. Saved BF16 weight views are replaced
by immutable NF4 descriptors, decoded on demand in backward and then freed.
"""
import os
import types
from pathlib import Path
import torch
from torch.nn import functional as F
from grouped_moe import prepare
from route_moe import route_indices

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent
        build = here / 'cpp-unique-scatter-round3-build'
        build.mkdir(exist_ok=True)
        os.environ.setdefault('MAX_JOBS', '1')
        _extension = load(name='solos_qwen_unique_scatter_round3', sources=[str(here/'expert_unique_scatter_round3.cpp')],
            build_directory=str(build), extra_cflags=['-O2', '-g0'], with_cuda=False, verbose=True)
    return _extension


class RecomputeExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ks, tokens, counts, packed, offload=False):
        # Retain only graph-local inputs, never reuse language activations across updates.
        ctx.x=x.detach();ctx.weights=weights.detach();ctx.ks=ks;ctx.tokens=tokens;ctx.counts=counts;ctx.packed=packed
        decoded={n:p.decode() for n,p in packed.items()}
        values=extension().expert_loop(x,weights,ks,tokens,counts,decoded['gate_proj'],decoded['up_proj'],decoded['down_proj'])
        ctx.activations=[t.detach().cpu() for t in values[1:]] if offload else values[1:]
        ctx.offload=offload
        return values[0]

    @staticmethod
    def backward(ctx, upstream):
        # Native autograd derivatives run outside autocast dispatch.
        with torch.autocast('cuda', enabled=False):
            decoded={n:p.decode() for n,p in ctx.packed.items()}
            gx,gw=extension().expert_backward(ctx.x,ctx.weights,ctx.ks,ctx.tokens,ctx.counts,
                decoded['gate_proj'],decoded['up_proj'],decoded['down_proj'],upstream,True,[t.to(ctx.x.device) for t in ctx.activations] if ctx.offload else ctx.activations)
        del ctx.x,ctx.weights,ctx.ks,ctx.tokens,ctx.counts,ctx.packed,ctx.activations
        return gx,gw,None,None,None,None,None


def forward(self, hidden_states):
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    logits = self.gate(x)
    weights = F.softmax(logits, dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)
    counts, ks, tokens = route_indices(selected, self.num_experts)
    out=RecomputeExperts.apply(x,weights,ks,tokens,counts,self._packed_nf4,getattr(self,'_probe_offload',False))
    return out.view(shape), logits


def install(qwen, share_storage=True, probe_offload=False):
    extension()
    blocks = [m for m in qwen.modules()
              if type(m).__name__ == 'Qwen3OmniMoeThinkerTextSparseMoeBlock']
    assert len(blocks) == 48
    for block in blocks:
        prepare(block, share_storage)
        block._probe_offload = probe_offload
        block.forward = types.MethodType(forward, block)
    return blocks
