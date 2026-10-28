"""Uninstalled v60 component: forward NF4 constant-codebook decode.
The packed/scales/row-pair/dot geometry and arithmetic order are reused.
No speed claim or GPU readiness. Bind real codebook only at a graph-free boundary.
"""
from dataclasses import dataclass
import math
import torch
import triton
import triton.language as tl
from nf4_row_scale_candidate_v44 import validate_table, _tensor_identity

@dataclass(frozen=True)
class BoundCodebook:
    values: tuple
    source: tuple

def prepare_codebook(p):
    if p.code.shape != (16,) or p.code.dtype != torch.float32 or p.code.requires_grad:
        raise ValueError("Require the actual frozen sixteen-entry FP32 codebook")
    source = _tensor_identity(p.code)
    values = tuple(p.code.detach().cpu().tolist())
    if len(values) != 16 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid codebook")
    if _tensor_identity(p.code) != source:
        raise RuntimeError("Source changed during boundary preparation")
    return BoundCodebook(values, source)

def validate_codebook(p, book):
    if type(book) is not BoundCodebook or _tensor_identity(p.code) != book.source:
        raise RuntimeError("Codebook source changed; rebuild at boundary")
    return book.values

@triton.jit
def _weight(DATA, SCALES, CODEBOOK: tl.constexpr, i, mask, BLOCK: tl.constexpr):
    q = tl.load(DATA + i // 2, mask, other=0).to(tl.int32)
    nibble = tl.where(i % 2 == 0, q >> 4, q & 15)
    scale = tl.load(SCALES + i // BLOCK, mask, other=0).to(tl.float32)
    code = tl.where(nibble < 8, tl.where(nibble < 4, tl.where(nibble < 2, tl.where(nibble < 1, CODEBOOK[0], CODEBOOK[1]), tl.where(nibble < 3, CODEBOOK[2], CODEBOOK[3])), tl.where(nibble < 6, tl.where(nibble < 5, CODEBOOK[4], CODEBOOK[5]), tl.where(nibble < 7, CODEBOOK[6], CODEBOOK[7]))), tl.where(nibble < 12, tl.where(nibble < 10, tl.where(nibble < 9, CODEBOOK[8], CODEBOOK[9]), tl.where(nibble < 11, CODEBOOK[10], CODEBOOK[11])), tl.where(nibble < 14, tl.where(nibble < 13, CODEBOOK[12], CODEBOOK[13]), tl.where(nibble < 15, CODEBOOK[14], CODEBOOK[15]))))
    weight = (code * scale).to(tl.bfloat16)
    return tl.where(mask, weight, 0.)


@triton.jit(do_not_specialize=['ROWS', 'TOTAL_ROWS'])
def _mm(A, DATA, SCALES, CODEBOOK: tl.constexpr, Ends, Y,
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
                w0 = _weight(DATA, SCALES, CODEBOOK,
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
                        w1 = _weight(DATA, SCALES, CODEBOOK,
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

def launch(a, p, table, ends, rows, transposed, geometry, *, codebook, write_once=False):
    if transposed is not True:
        raise ValueError("This candidate applies to forward projections only")
    values = validate_codebook(p, codebook)
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
    _mm[(ctas,)](a, p.data, scales, values, ends, y, n, k, p.experts, rows,
                p.shape[0] * p.shape[1], transposed, p.blocksize,
                p.scale_state.blocksize, bn, bk, ctas, a.shape[0], write_once,
                num_warps=geometry['warps'], num_stages=geometry['stages'],
                enable_fp_fusion=False)
    return y
