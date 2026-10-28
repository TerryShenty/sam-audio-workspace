"""Isolated route aggregation draft; not installed or CPU/GPU validated.

Elides full padded-row pointwise temporaries without changing expert GEMMs,
top-k selection, route order, or BF16 rounding boundaries. No tensor cache.
"""
import weakref
import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_sum(Y, W, INDICES, SLOTS, OUT, N: tl.constexpr,
                  D: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = off // D, off % D
    ok = token < N
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in range(K):
        row = tl.load(INDICES + token * K + j, ok, 0)
        slot = tl.load(SLOTS + token * K + j, ok, 0)
        y = tl.load(Y + row * D + col, ok, 0).to(tl.float32)
        w = tl.load(W + token * K + slot, ok, 0).to(tl.float32)
        # Original y*w materializes BF16, then each ordered add rounds BF16.
        value = (y * w).to(OUT.dtype.element_ty).to(tl.float32)
        acc = (acc + value).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + off, acc, ok)


@triton.jit
def _sum_two_reverse(G, U, INDICES, OUT, N: tl.constexpr,
                     D: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = off // D, off % D
    ok = token < N
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in range(K):
        row = tl.load(INDICES + token * K + K - 1 - j, ok, 0)
        g = tl.load(G + row * D + col, ok, 0).to(tl.float32)
        u = tl.load(U + row * D + col, ok, 0).to(tl.float32)
        # Original gate_dx+up_dx rounds before reverse expert reduction.
        value = (g + u).to(OUT.dtype.element_ty).to(tl.float32)
        acc = (acc + value).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + off, acc, ok)


def eligible(values, indices, *, weights=None, slots=None):
    if not values or indices.ndim != 2 or not indices.is_cuda or not indices.is_contiguous():
        return False
    n, k = indices.shape
    if k < 1 or k > 16 or indices.dtype not in (torch.int32, torch.int64):
        return False
    shape = values[0].shape
    if len(shape) != 2 or not all(v.is_cuda and v.dtype == torch.bfloat16 and
            v.is_contiguous() and v.device == indices.device and v.shape == shape
            for v in values):
        return False
    if weights is not None and not (weights.shape == indices.shape and weights.is_contiguous()
            and weights.device == indices.device and weights.dtype == torch.bfloat16):
        return False
    if slots is not None and not (slots.shape == indices.shape and slots.is_contiguous()
            and slots.device == indices.device and slots.dtype in (torch.int32, torch.int64)):
        return False
    # Route index bounds/uniqueness come from the unchanged routing construction;
    # no host synchronization or value-dependent scan is introduced here.
    return True


class RouteAggregatePolicy:
    def __init__(self):
        self.mode, self.block = 'disabled', 256
        self._live = weakref.WeakSet()
        self.counts = dict(forward_calls=0, backward_calls=0,
                          forward_fallbacks=0, backward_fallbacks=0,
                          elided_pointwise_tensor_bytes=0)

    def boundary(self):
        if self._live:
            raise RuntimeError('Cannot reconfigure route aggregation with live graphs')

    def configure(self, mode='disabled', block=256):
        self.boundary()
        if mode not in ('disabled', 'weighted_forward', 'input_backward', 'both') or \
                type(block) is not int or block not in (256, 512, 1024):
            raise ValueError('Unsupported route aggregation policy')
        self.mode, self.block = mode, block

    def attach(self, ctx):
        if self.mode == 'disabled':
            return None
        ctx.route_aggregate_mode, ctx.route_aggregate_block = self.mode, self.block
        self._live.add(ctx)
        return self

    def forward(self, ctx, y, weights, indices, slots):
        if ctx.route_aggregate_mode not in ('weighted_forward', 'both'):
            return None
        if not eligible((y,), indices, weights=weights, slots=slots):
            self.counts['forward_fallbacks'] += 1
            return None
        n, k = indices.shape
        d, block = y.shape[1], ctx.route_aggregate_block
        out = y.new_empty(n, d)
        if n * d:
            _weighted_sum[(triton.cdiv(n * d, block),)](
                y, weights, indices, slots, out, n, d, k, block,
                num_warps=4, enable_fp_fusion=False)
        self.counts['forward_calls'] += 1
        self.counts['elided_pointwise_tensor_bytes'] += y.numel() * y.element_size()
        return out

    def backward(self, ctx, gate_dx, up_dx, indices):
        if ctx.route_aggregate_mode not in ('input_backward', 'both'):
            return None
        if not eligible((gate_dx, up_dx), indices):
            self.counts['backward_fallbacks'] += 1
            return None
        n, k = indices.shape
        d, block = gate_dx.shape[1], ctx.route_aggregate_block
        out = gate_dx.new_empty(n, d)
        if n * d:
            _sum_two_reverse[(triton.cdiv(n * d, block),)](
                gate_dx, up_dx, indices, out, n, d, k, block,
                num_warps=4, enable_fp_fusion=False)
        self.counts['backward_calls'] += 1
        self.counts['elided_pointwise_tensor_bytes'] += gate_dx.numel() * gate_dx.element_size()
        return out

    def release(self, ctx):
        self._live.discard(ctx)

    def statistics(self):
        return dict(self.counts, mode=self.mode, block=self.block,
                    live_graphs=len(self._live), gpu_validated=False,
                    structural_bytes_are_not_peak_reduction=True)
