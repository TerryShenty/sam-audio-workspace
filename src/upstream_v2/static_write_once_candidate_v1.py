"""Isolated static NF4 output-initialization candidate; never self-installs.

Preserve active-tile arithmetic and scheduling. Inactive capacity tiles and
the trailing sentinel rows are zeroed by the same kernel. Every output cell
is written once, permitting new_empty instead of a separate full-buffer fill.
Only static row32 is eligible. Dynamic/long-input fallback remains original.
CPU interpreter/codegen checks are not GPU numeric or throughput acceptance.
"""
import torch
import triton
import triton.language as tl
from projection_backend_candidate_v25 import packed_roles
from projection_launch_candidate_v25 import select


@triton.jit(do_not_specialize=["ROWS", "MTILES", "TOTAL_ROWS"])
def _mm(A, DATA, ABS, ABS2, CODE2, OFFSETS, CODE4, Ends, O,
        N: tl.constexpr, K: tl.constexpr, E: tl.constexpr, ROWS,
        PER: tl.constexpr, TRANSPOSED: tl.constexpr, BLOCK: tl.constexpr,
        SCALE_BLOCK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr, CTAS: tl.constexpr, MTILES, GROUP: tl.constexpr,
        TOTAL_ROWS, ZERO_BLOCK: tl.constexpr):
    NCOLS = tl.cdiv(N, BN)
    for linear in range(tl.program_id(0), MTILES * NCOLS, CTAS):
        group_id = linear // (GROUP * NCOLS)
        first = group_id * GROUP
        group_size = tl.minimum(MTILES - first, GROUP)
        tile = first + (linear % (GROUP * NCOLS)) % group_size
        column = (linear % (GROUP * NCOLS)) // group_size
        rows = tl.load(Ends + E - 1)
        r = tile * BM + tl.arange(0, BM)
        c = column * BN + tl.arange(0, BN)
        if tile * BM < rows:
            ids = tl.arange(0, triton.next_power_of_2(E))
            ends = tl.load(Ends + ids, ids < E, other=2147483647)
            expert = tl.sum((tile * BM >= ends).to(tl.int32), 0)
            k = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + k
                aa = tl.load(A + r[:, None] * K + kk[None, :],
                             (r[:, None] < rows) & (kk[None, :] < K), other=0)
                if TRANSPOSED:
                    i = expert * PER + c[None, :] * K + kk[:, None]
                else:
                    i = expert * PER + kk[:, None] * N + c[None, :]
                valid = (kk[:, None] < K) & (c[None, :] < N)
                q = tl.load(DATA + i // 2, valid, other=0).to(tl.int32)
                nibble = tl.where(i % 2 == 0, q >> 4, q & 15)
                si = i // BLOCK
                sc = tl.load(ABS + si, valid, other=0).to(tl.int32)
                scale = tl.load(CODE2 + sc) * tl.load(ABS2 + si // SCALE_BLOCK, valid, other=0)
                scale = scale + tl.load(OFFSETS + expert)
                weight = (tl.load(CODE4 + nibble) * scale).to(tl.bfloat16)
                weight = tl.where(valid, weight, 0.)
                acc = tl.dot(aa, weight, acc)
            tl.store(O + r[:, None] * N + c[None, :], acc,
                     (r[:, None] < rows) & (c[None, :] < N))
        else:
            tl.store(O + r[:, None] * N + c[None, :], 0.,
                     (r[:, None] < ROWS) & (c[None, :] < N))
    for start in range(tl.program_id(0) * ZERO_BLOCK,
                       (TOTAL_ROWS - ROWS) * N, CTAS * ZERO_BLOCK):
        j = start + tl.arange(0, ZERO_BLOCK)
        tl.store(O + ROWS * N + j, 0., j < (TOTAL_ROWS - ROWS) * N)


def enabled(config):
    value = config.get('nf4_static_write_once', False)
    if type(value) is not bool:
        raise ValueError('Explicit boolean initialization policy required')
    if value and (not config.get('static', False) or config.get('nf4_row_tile', 32) != 32):
        raise ValueError('Write-once candidate requires static row32')
    return value


def identity_suffix(config):
    return ('nf4_static_write_once_v1',) if enabled(config) else ()


def install(module, choose_config, blocks, *, memo=None):
    """Install after projection dispatch, before memo, in a future isolated worker.

    Caller must add identity_suffix to full identity/compiler variant and
    continuation manifests. This component alone is not a worker integration.
    """
    if module.BM != 32 or (memo is not None and memo.live_bytes):
        raise RuntimeError('Require row32 and no live memo graph')
    roles = packed_roles(blocks)
    original = module.grouped_mm

    def grouped_mm(a, p, ends, rows, transposed):
        cfg = choose_config()
        if not enabled(cfg):
            return original(a, p, ends, rows, transposed)
        entry = roles.get(id(p))
        if entry is None or entry[0] is not p:
            raise ValueError('Unknown packed identity')
        v = select(cfg, entry[1], transposed)
        n, k = p.shape if transposed else p.shape[::-1]
        if not (a.is_contiguous() and a.dtype == torch.bfloat16 and a.is_cuda and
                p.sources is None and a.ndim == 2 and a.shape[1] == k and
                rows > 0 and rows % 32 == 0 and a.shape[0] >= rows and
                ends.is_contiguous() and ends.dtype == torch.int32 and
                ends.device == a.device and ends.numel() == p.experts):
            raise ValueError('Invalid static kernel inputs')
        output = a.new_empty(a.shape[0], n)
        ctas = min(rows // 32 * triton.cdiv(n, v['bn']),
                   torch.cuda.get_device_properties(a.device).multi_processor_count * v['waves'])
        _mm[(ctas,)](a, p.data, p.absmax, p.scale_state.absmax, p.scale_state.code,
                     p.offsets, p.code, ends, output, n, k, p.experts, rows,
                     p.shape[0] * p.shape[1], transposed, p.blocksize,
                     p.scale_state.blocksize, 32, v['bn'], v['bk'], ctas,
                     rows // 32, v['group'], a.shape[0], 256,
                     num_warps=v['warps'], num_stages=v['stages'], enable_fp_fusion=False)
        return output

    module.grouped_mm = grouped_mm

    def undo():
        if module.grouped_mm is not grouped_mm or (memo is not None and memo.live_bytes):
            raise RuntimeError('Remove outer wrappers and release graphs before undo')
        module.grouped_mm = original
    return undo
