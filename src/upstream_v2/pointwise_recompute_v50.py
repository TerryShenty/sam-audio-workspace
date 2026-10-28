"""Isolated SiLU-product kernels. No self-installation or cross-update tensors."""
import weakref
import torch
import triton
import triton.language as tl

@triton.jit
def _forward(G, U, M, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    g = tl.load(G + i, valid, 0).to(tl.float32)
    u = tl.load(U + i, valid, 0).to(tl.float32)
    # Match the materialized BF16 SiLU before the original BF16 product.
    a = (g / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(M + i, a * u, valid)

@triton.jit
def _backward(DM, U, G, DG, DU, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    dm = tl.load(DM + i, valid, 0).to(tl.float32)
    u = tl.load(U + i, valid, 0).to(tl.float32)
    g = tl.load(G + i, valid, 0).to(tl.float32)
    sig = 1. / (1. + tl.exp(-g))
    a = (g / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    da = (dm * u).to(tl.bfloat16).to(tl.float32)
    tl.store(DG + i, da * (sig * (1. + g * (1. - sig))), valid)
    tl.store(DU + i, dm * a, valid)

def eligible(*values):
    return bool(values) and all(v.is_cuda and v.dtype == torch.bfloat16 and
        v.is_contiguous() and v.shape == values[0].shape and
        v.device == values[0].device for v in values)

def validate(values, block):
    if type(block) is not int or block not in (256, 512, 1024):
        raise ValueError('Unsupported pointwise block')
    if not eligible(*values):
        raise ValueError('Pointwise kernels require matching contiguous CUDA BF16 tensors')

def forward(g, u, block=256):
    validate((g, u), block)
    m = torch.empty_like(g)
    if g.numel():
        _forward[(triton.cdiv(g.numel(), block),)](g, u, m, g.numel(), block,
            num_warps=4, enable_fp_fusion=False)
    return m

def backward(dm, u, g, block=256):
    validate((dm, u, g), block)
    dg, du = torch.empty_like(g), torch.empty_like(u)
    if g.numel():
        _backward[(triton.cdiv(g.numel(), block),)](dm, u, g, dg, du,
            g.numel(), block, num_warps=4, enable_fp_fusion=False)
    return dg, du

class PointwisePolicy:
    """Disabled by default. Counts dispatches; does not claim GPU validation."""
    def __init__(self):
        self.mode, self.block = 'disabled', 256
        self._live = weakref.WeakSet()
        self.counts = dict(forward_calls=0, backward_calls=0,
            fallback_calls=0, elided_saved_activation_bytes=0)

    def boundary(self):
        if self._live:
            raise RuntimeError('Cannot change pointwise policy with a live graph')

    def configure(self, mode='disabled', block=256):
        self.boundary()
        if mode not in ('disabled', 'recompute') or type(block) is not int or block not in (256, 512, 1024):
            raise ValueError('Unsupported pointwise policy')
        self.mode, self.block = mode, block

    def select(self, g, u):
        if self.mode == 'disabled':
            return False
        if not eligible(g, u):
            self.counts['fallback_calls'] += 1
            return False
        return True

    def forward(self, ctx, g, u):
        ctx.pointwise_block = self.block
        self._live.add(ctx)
        m = forward(g, u, self.block)
        self.counts['forward_calls'] += 1
        self.counts['elided_saved_activation_bytes'] += g.numel() * g.element_size()
        return m

    def backward(self, ctx, dm, u, g):
        out = backward(dm, u, g, ctx.pointwise_block)
        self.counts['backward_calls'] += 1
        return out

    def release(self, ctx):
        self._live.discard(ctx)

    def statistics(self):
        return dict(self.counts, mode=self.mode, block=self.block, live_graphs=len(self._live),
            gpu_validated=False)
