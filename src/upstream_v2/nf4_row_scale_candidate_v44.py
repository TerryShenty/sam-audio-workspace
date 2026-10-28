"""Isolated scale-metadata + persistent row-pair component, not installed.

Two scale storage precisions (FP32/BF16) combine with the current two-row32
persistent CTA traversal. This is not the old round21 single-row direct grid.
Only immutable quantization metadata is retained; no decoded full weights or
activation state. Preparation time, table bytes, finite GPU training, memory,
identity/restore and complete-step performance still need runtime integration.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl

@triton.jit
def _weight(DATA, SCALES, CODE4, i, mask, BLOCK: tl.constexpr):
    q = tl.load(DATA + i // 2, mask, other=0).to(tl.int32)
    nibble = tl.where(i % 2 == 0, q >> 4, q & 15)
    scale = tl.load(SCALES + i // BLOCK, mask, other=0).to(tl.float32)
    weight = (tl.load(CODE4 + nibble) * scale).to(tl.bfloat16)
    return tl.where(mask, weight, 0.)

@triton.jit
def _scales(ABS, ABS2, CODE2, OFFSETS, OUT, COUNT: tl.constexpr,
            PER_SCALES: tl.constexpr, SCALE_BLOCK: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < COUNT
    code = tl.load(ABS + i, valid, other=0).to(tl.int32)
    value = tl.load(CODE2 + code) * tl.load(ABS2 + i // SCALE_BLOCK, valid, other=0)
    value += tl.load(OFFSETS + i // PER_SCALES, valid, other=0)
    tl.store(OUT + i, value, valid)

@triton.jit(do_not_specialize=['ROWS', 'TOTAL_ROWS'])
def _mm(A, DATA, SCALES, CODE4, Ends, Y,
        N: tl.constexpr, K: tl.constexpr, E: tl.constexpr, ROWS,
        PER: tl.constexpr, TRANSPOSED: tl.constexpr, BLOCK: tl.constexpr,
        SCALE_BLOCK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        CTAS: tl.constexpr, TOTAL_ROWS, WRITE_ONCE: tl.constexpr):
    cols = tl.cdiv(N, BN)
    row_pairs = tl.cdiv(ROWS, 64)
    active = tl.load(Ends + E - 1)
    ids = tl.arange(0, triton.next_power_of_2(E))
    ends = tl.load(Ends + ids, ids < E, other=2147483647)
    for linear in range(tl.program_id(0), row_pairs * cols, CTAS):
        first = (linear // cols) * 64
        r0 = first + tl.arange(0, 32)
        r1 = first + 32 + tl.arange(0, 32)
        c = (linear % cols) * BN + tl.arange(0, BN)
        if first < active:
            e0 = tl.sum((first >= ends).to(tl.int32), 0)
            e1 = tl.minimum(tl.sum((first + 32 >= ends).to(tl.int32), 0), E - 1)
            acc0 = tl.zeros((32, BN), tl.float32)
            acc1 = tl.zeros((32, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + tl.arange(0, BK)
                mask = (kk[:, None] < K) & (c[None, :] < N)
                if TRANSPOSED:
                    index = c[None, :] * K + kk[:, None]
                else:
                    index = kk[:, None] * N + c[None, :]
                w0 = _weight(DATA, SCALES, CODE4,
                             e0 * PER + index, mask, BLOCK)
                a0 = tl.load(A + r0[:, None] * K + kk[None, :],
                             (r0[:, None] < active) & (kk[None, :] < K), other=0)
                acc0 = tl.dot(a0, w0, acc0)
                if first + 32 < active:
                    a1 = tl.load(A + r1[:, None] * K + kk[None, :],
                                 (r1[:, None] < active) & (kk[None, :] < K), other=0)
                    if e0 == e1:
                        acc1 = tl.dot(a1, w0, acc1)
                    else:
                        w1 = _weight(DATA, SCALES, CODE4,
                                     e1 * PER + index, mask, BLOCK)
                        acc1 = tl.dot(a1, w1, acc1)
            tl.store(Y + r0[:, None] * N + c[None, :], acc0,
                     (r0[:, None] < active) & (c[None, :] < N))
            if first + 32 < active:
                tl.store(Y + r1[:, None] * N + c[None, :], acc1,
                         (r1[:, None] < active) & (c[None, :] < N))
            elif WRITE_ONCE:
                tl.store(Y + r1[:, None] * N + c[None, :], 0.,
                         (r1[:, None] < ROWS) & (c[None, :] < N))
        elif WRITE_ONCE:
            tl.store(Y + r0[:, None] * N + c[None, :], 0.,
                     (r0[:, None] < ROWS) & (c[None, :] < N))
            tl.store(Y + r1[:, None] * N + c[None, :], 0.,
                     (r1[:, None] < ROWS) & (c[None, :] < N))
    if WRITE_ONCE:
        for offset in range(tl.program_id(0) * 256, (TOTAL_ROWS - ROWS) * N, CTAS * 256):
            j = offset + tl.arange(0, 256)
            tl.store(Y + ROWS * N + j, 0., j < (TOTAL_ROWS - ROWS) * N)


def _tensor_identity(t):
    if t.requires_grad:
        raise ValueError('Scale cache requires immutable frozen metadata')
    try:
        version = t._version
    except RuntimeError as error:
        raise ValueError('Metadata must expose mutation versions') from error
    return (id(t), t.data_ptr(), version, tuple(t.shape), tuple(t.stride()),
            str(t.dtype), str(t.device))


def source_identity(p):
    if p.sources is not None or len(p.shape) != 2 or any(v <= 0 for v in p.shape):
        raise ValueError('Static packed weights required')
    per = p.shape[0] * p.shape[1]
    if p.experts <= 0 or p.blocksize <= 0 or per % p.blocksize or p.scale_state.blocksize <= 0:
        raise ValueError('Expert weights must align to quantization blocks')
    tensors = (p.data, p.absmax, p.scale_state.absmax, p.scale_state.code, p.offsets, p.code)
    if any(t.device != p.data.device or not t.is_contiguous() for t in tensors):
        raise ValueError('Packed metadata must be contiguous on one device')
    count = p.experts * per // p.blocksize
    if (p.data.numel() != p.experts * per // 2 or p.data.dtype != torch.uint8
            or p.absmax.numel() != count or p.absmax.dtype != torch.uint8
            or p.scale_state.absmax.numel() < triton.cdiv(count, p.scale_state.blocksize)
            or p.scale_state.code.numel() != 256 or p.code.numel() != 16
            or p.offsets.numel() != p.experts):
        raise ValueError('Packed scale metadata shape mismatch')
    if any(t.dtype != torch.float32 for t in tensors[2:]):
        raise ValueError('Nested scale and codebook metadata must be FP32')
    return (id(p), tuple(p.shape), p.experts, p.blocksize, p.scale_state.blocksize,
            tuple(_tensor_identity(t) for t in tensors))


@dataclass(frozen=True)
class ScaleTable:
    values: torch.Tensor
    source: tuple
    payload: tuple

    @property
    def nbytes(self):
        return self.values.numel() * self.values.element_size()


def _bind_table(p, values):
    identity = source_identity(p)
    count = p.experts * p.shape[0] * p.shape[1] // p.blocksize
    if (values.dtype not in (torch.float32, torch.bfloat16) or values.ndim != 1
            or values.numel() != count or not values.is_contiguous()
            or values.device != p.data.device):
        raise ValueError('Scale table shape, precision or device mismatch')
    return ScaleTable(values, identity, _tensor_identity(values))


def validate_table(p, table):
    if not isinstance(table, ScaleTable) or source_identity(p) != table.source:
        raise ValueError('Scale table belongs to another or modified source')
    if _tensor_identity(table.values) != table.payload:
        raise ValueError('Scale table payload changed')
    return table.values


def prepare_scales(p, dtype=torch.float32):
    source_identity(p)
    if not p.data.is_cuda or dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Explicit GPU metadata preparation with FP32/BF16 storage required')
    per_scales = p.shape[0] * p.shape[1] // p.blocksize
    count = p.experts * per_scales
    values = torch.empty(count, device=p.data.device, dtype=dtype)
    _scales[(triton.cdiv(count, 256),)](
        p.absmax, p.scale_state.absmax, p.scale_state.code, p.offsets, values,
        count, per_scales, p.scale_state.blocksize, 256, enable_fp_fusion=False)
    return _bind_table(p, values)


def launch(a, p, table, ends, rows, transposed, geometry, *, write_once=False):
    scales = validate_table(p, table)
    n, k = p.shape if transposed else p.shape[::-1]
    if not (a.is_cuda and a.dtype == torch.bfloat16 and a.is_contiguous()
            and a.ndim == 2 and a.shape[1] == k and a.shape[0] >= rows > 0
            and rows % 32 == 0 and ends.is_contiguous()
            and ends.dtype == torch.int32 and ends.numel() == p.experts
            and ends.device == a.device == p.data.device):
        raise ValueError('Require static row32 BF16 GPU inputs')
    if (geometry['bn'] not in (64, 128) or geometry['bk'] != 32
            or geometry['warps'] != 4 or geometry['stages'] != 1
            or geometry['waves'] not in (2, 4, 8, 16, 32)):
        raise ValueError('Only prepared two-row BK32 geometry')
    bn, bk = geometry['bn'], geometry['bk']
    ctas = min(triton.cdiv(rows, 64) * triton.cdiv(n, bn),
               torch.cuda.get_device_properties(a.device).multi_processor_count * geometry['waves'])
    y = (a.new_empty if write_once else a.new_zeros)(a.shape[0], n)
    _mm[(ctas,)](a, p.data, scales, p.code, ends, y, n, k, p.experts, rows,
                p.shape[0] * p.shape[1], transposed, p.blocksize,
                p.scale_state.blocksize, bn, bk, ctas, a.shape[0], write_once,
                num_warps=geometry['warps'], num_stages=geometry['stages'],
                enable_fp_fusion=False)
    return y
