"""Research candidate: expose graph activations to PyTorch checkpoint hooks.

Same C++ expert forward/backward as production. Tensor saves use the documented
autograd API, so non-reentrant checkpointing can discard and reconstruct them.
Not installed into production by importing this module.
"""
import torch
from torch.nn import functional as F
from route_moe import route_indices
from bulk_gather_support_round6 import extension, decode


class CheckpointAwareExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ks, tokens, counts, packed, offload=False):
        ctx.counts, ctx.packed, ctx.offload = counts, packed, offload
        decoded = decode(packed)
        values = extension().expert_loop(x, weights, ks, tokens, counts,
            decoded['gate_proj'], decoded['up_proj'], decoded['down_proj'])
        activations = [t.detach().cpu() for t in values[1:]] if offload else values[1:]
        ctx.save_for_backward(x, weights, ks, tokens, *activations)
        return values[0]

    @staticmethod
    def backward(ctx, upstream):
        x, weights, ks, tokens, *activations = ctx.saved_tensors
        with torch.autocast('cuda', enabled=False):
            decoded = decode(ctx.packed)
            gx, gw = extension().expert_backward(x, weights, ks, tokens,
                ctx.counts, decoded['gate_proj'], decoded['up_proj'],
                decoded['down_proj'], upstream, True,
                [t.to(x.device) for t in activations] if ctx.offload else activations)
        return gx, gw, None, None, None, None, None


def forward(self, hidden_states):
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    logits = self.gate(x)
    weights = F.softmax(logits, dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    counts, ks, tokens = route_indices(selected, self.num_experts)
    out = CheckpointAwareExperts.apply(x, weights.to(x.dtype), ks, tokens,
        counts, self._packed_nf4, getattr(self, '_probe_offload', False))
    return out.view(shape), logits
