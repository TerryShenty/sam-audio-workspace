"""Candidate NF4 MoE execution, not automatically installed into training.

Immutable NF4 bytes retain their original quantization. Quantized scales stay quantized. Existing NF4 storage may be reused without
repacking all 48 layers persistently. BF16 expert weights are temporary
per grouped matmul, including backward; no language activations are cached.
"""
import types
import torch
from torch.nn import functional as F
import bitsandbytes.functional as BF


class PackedWeight:
    def __init__(self, linears, share_storage=False):
        states = [m.weight.quant_state for m in linears]
        first = states[0]
        assert first.quant_type == 'nf4' and first.dtype == torch.bfloat16
        assert all(not m.weight.requires_grad and m.bias is None for m in linears)
        assert all(s.shape == first.shape and s.blocksize == first.blocksize
                   and s.dtype == first.dtype and s.quant_type == first.quant_type for s in states)
        self.experts = len(linears)
        self.shape = tuple(first.shape)
        # Reuse immutable original storage; a layer-sized concatenation is cheap
        # and avoids fragmentation from repacking all 48 layers simultaneously.
        self.sources = [m.weight for m in linears] if share_storage else None
        self.data = None if share_storage else torch.cat([m.weight.detach().reshape(-1) for m in linears]).view(-1, 1)
        assert all(s.nested and not s.state2.nested for s in states)
        assert all(s.state2.blocksize == first.state2.blocksize
                   and torch.equal(s.state2.code, first.state2.code) for s in states)
        self.absmax = torch.cat([s.absmax for s in states])
        self.offsets = torch.stack([s.offset for s in states]).view(self.experts, 1)
        self.scale_state = BF.QuantState(absmax=torch.cat([s.state2.absmax for s in states]),
            code=first.state2.code, blocksize=first.state2.blocksize,
            dtype=first.state2.dtype)
        self.blocksize = first.blocksize
        self.code = first.code

    def decode(self):
        data = self.data if self.sources is None else torch.cat([w.detach().reshape(-1) for w in self.sources]).view(-1, 1)
        scales = BF.dequantize_blockwise(self.absmax, self.scale_state)
        scales = (scales.view(self.experts, -1) + self.offsets).flatten()
        state = BF.QuantState(absmax=scales,
            shape=(self.experts * self.shape[0], self.shape[1]), dtype=torch.bfloat16,
            blocksize=self.blocksize, quant_type='nf4', code=self.code)
        return BF.dequantize_4bit(data, state).view(self.experts, *self.shape)


class PackedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed):
        assert x.dtype == torch.bfloat16
        ctx.packed = packed
        return torch.bmm(x, packed.decode().transpose(1, 2))

    @staticmethod
    def backward(ctx, grad):
        # Frozen weights: only the input derivative is required, as in bnb.
        return torch.bmm(grad, ctx.packed.decode()), None


def prepare(block, share_storage=False):
    block._packed_nf4 = {
        name: PackedWeight([getattr(e, name) for e in block.experts], share_storage)
        for name in ('gate_proj', 'up_proj', 'down_proj')}
    return block


def forward(self, hidden_states):
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    n = x.shape[0]
    logits = self.gate(x)
    weights = F.softmax(logits, dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)
    # Stable top-k-major routing matches the original torch.where ordering.
    flat = selected.transpose(0, 1).reshape(-1)
    order = torch.argsort(flat, stable=True)
    expert = flat[order]
    counts = torch.bincount(flat, minlength=self.num_experts)
    capacity = ((int(counts.max()) + 15) // 16) * 16
    starts = counts.cumsum(0) - counts
    slot = torch.arange(order.numel(), device=x.device) - starts[expert]
    destination = expert * capacity + slot
    routed = x.new_zeros(self.num_experts * capacity, x.shape[1])
    routed = routed.index_copy(0, destination, x[order.remainder(n)])
    routed = routed.view(self.num_experts, capacity, x.shape[1])
    g = PackedLinear.apply(routed, self._packed_nf4['gate_proj'])
    u = PackedLinear.apply(routed, self._packed_nf4['up_proj'])
    y = PackedLinear.apply(self.experts[0].act_fn(g) * u, self._packed_nf4['down_proj'])
    routed_y = y.reshape(-1, x.shape[1])[destination]
    # Unroute, then add the eight BF16 contributions in increasing expert order.
    # A sum/atomic scatter would change the original BF16 addition order.
    ordered = torch.empty_like(routed_y).index_copy(0, order, routed_y)
    ordered = ordered.view(self.top_k, n, -1).transpose(0, 1)
    ordered = ordered * weights.unsqueeze(-1)
    expert_order = selected.argsort(dim=1)
    ordered = ordered.gather(1, expert_order.unsqueeze(-1).expand_as(ordered))
    out = torch.zeros_like(x)
    for k in range(self.top_k):
        out = out + ordered[:, k]
    return out.view(shape), logits


def install(qwen, share_storage=True):
    blocks = [m for m in qwen.modules()
              if type(m).__name__ == 'Qwen3OmniMoeThinkerTextSparseMoeBlock']
    assert len(blocks) == 48
    for block in blocks:
        prepare(block, share_storage)
        block.forward = types.MethodType(forward, block)
    return blocks
