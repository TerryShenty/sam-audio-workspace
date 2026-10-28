"""Independent four-row32 local NF4 decode reuse candidate.

No global or cross-update weight cache. Same-expert reuse is confined to the
current K tile; row32 routing and output shape remain unchanged. Not integrated
into the running v32 worker; only affected CPU checks and offline compilation
are preparation evidence, never GPU speed or memory acceptance.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _weight(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4, i, mask, expert,
            BLOCK: tl.constexpr, SCALE_BLOCK: tl.constexpr):
    q = tl.load(DATA + i // 2, mask, other=0).to(tl.int32)
    nibble = tl.where(i % 2 == 0, q >> 4, q & 15)
    si = i // BLOCK
    code = tl.load(ABS + si, mask, other=0).to(tl.int32)
    scale = tl.load(CODE2 + code) * tl.load(ABS2 + si // SCALE_BLOCK, mask, other=0)
    scale += tl.load(OFFSETS + expert)
    weight = (tl.load(CODE4 + nibble) * scale).to(tl.bfloat16)
    return tl.where(mask, weight, 0.)


@triton.jit(do_not_specialize=['ROWS', 'TOTAL_ROWS'])
def _mm(A, DATA, ABS, ABS2, CODE2, OFFSETS, CODE4, Ends, Y,
        N: tl.constexpr, K: tl.constexpr, E: tl.constexpr, ROWS,
        PER: tl.constexpr, TRANSPOSED: tl.constexpr, BLOCK: tl.constexpr,
        SCALE_BLOCK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        CTAS: tl.constexpr, TOTAL_ROWS, WRITE_ONCE: tl.constexpr):
    cols = tl.cdiv(N, BN)
    row_pairs = tl.cdiv(ROWS, 128)
    active = tl.load(Ends + E - 1)
    ids = tl.arange(0, triton.next_power_of_2(E))
    ends = tl.load(Ends + ids, ids < E, other=2147483647)
    for linear in range(tl.program_id(0), row_pairs * cols, CTAS):
        first = (linear // cols) * 128
        r0 = first + tl.arange(0, 32)
        r1 = first + 32 + tl.arange(0, 32)
        r2 = first + 64 + tl.arange(0, 32)
        r3 = first + 96 + tl.arange(0, 32)
        c = (linear % cols) * BN + tl.arange(0, BN)
        if first < active:
            e0 = tl.sum((first >= ends).to(tl.int32), 0)
            e1 = tl.minimum(tl.sum((first + 32 >= ends).to(tl.int32), 0), E - 1)
            e2 = tl.minimum(tl.sum((first + 64 >= ends).to(tl.int32), 0), E - 1)
            e3 = tl.minimum(tl.sum((first + 96 >= ends).to(tl.int32), 0), E - 1)
            acc0 = tl.zeros((32, BN), tl.float32)
            acc1 = tl.zeros((32, BN), tl.float32)
            acc2 = tl.zeros((32, BN), tl.float32)
            acc3 = tl.zeros((32, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + tl.arange(0, BK)
                mask = (kk[:, None] < K) & (c[None, :] < N)
                if TRANSPOSED:
                    index = c[None, :] * K + kk[:, None]
                else:
                    index = kk[:, None] * N + c[None, :]
                w0 = _weight(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4,
                             e0 * PER + index, mask, e0, BLOCK, SCALE_BLOCK)
                a0 = tl.load(A + r0[:, None] * K + kk[None, :],
                             (r0[:, None] < active) & (kk[None, :] < K), other=0)
                acc0 = tl.dot(a0, w0, acc0)
                w1 = w0
                if first + 32 < active:
                    a1 = tl.load(A + r1[:, None] * K + kk[None, :],
                                 (r1[:, None] < active) & (kk[None, :] < K), other=0)
                    if e1 != e0:
                        w1 = _weight(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4,
                                     e1 * PER + index, mask, e1, BLOCK, SCALE_BLOCK)
                    acc1 = tl.dot(a1, w1, acc1)
                w2 = w1
                if first + 64 < active:
                    a2 = tl.load(A + r2[:, None] * K + kk[None, :],
                                 (r2[:, None] < active) & (kk[None, :] < K), other=0)
                    if e2 != e1:
                        w2 = _weight(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4,
                                     e2 * PER + index, mask, e2, BLOCK, SCALE_BLOCK)
                    acc2 = tl.dot(a2, w2, acc2)
                w3 = w2
                if first + 96 < active:
                    a3 = tl.load(A + r3[:, None] * K + kk[None, :],
                                 (r3[:, None] < active) & (kk[None, :] < K), other=0)
                    if e3 != e2:
                        w3 = _weight(DATA, ABS, ABS2, CODE2, OFFSETS, CODE4,
                                     e3 * PER + index, mask, e3, BLOCK, SCALE_BLOCK)
                    acc3 = tl.dot(a3, w3, acc3)
            tl.store(Y + r0[:, None] * N + c[None, :], acc0,
                     (r0[:, None] < active) & (c[None, :] < N))
            if first + 32 < active:
                tl.store(Y + r1[:, None] * N + c[None, :], acc1,
                         (r1[:, None] < active) & (c[None, :] < N))
            elif WRITE_ONCE:
                tl.store(Y + r1[:, None] * N + c[None, :], 0.,
                         (r1[:, None] < ROWS) & (c[None, :] < N))
            if first + 64 < active:
                tl.store(Y + r2[:, None] * N + c[None, :], acc2,
                         (r2[:, None] < active) & (c[None, :] < N))
            elif WRITE_ONCE:
                tl.store(Y + r2[:, None] * N + c[None, :], 0.,
                         (r2[:, None] < ROWS) & (c[None, :] < N))
            if first + 96 < active:
                tl.store(Y + r3[:, None] * N + c[None, :], acc3,
                         (r3[:, None] < active) & (c[None, :] < N))
            elif WRITE_ONCE:
                tl.store(Y + r3[:, None] * N + c[None, :], 0.,
                         (r3[:, None] < ROWS) & (c[None, :] < N))
        elif WRITE_ONCE:
            tl.store(Y + r0[:, None] * N + c[None, :], 0.,
                     (r0[:, None] < ROWS) & (c[None, :] < N))
            tl.store(Y + r1[:, None] * N + c[None, :], 0.,
                     (r1[:, None] < ROWS) & (c[None, :] < N))
            tl.store(Y + r2[:, None] * N + c[None, :], 0.,
                     (r2[:, None] < ROWS) & (c[None, :] < N))
            tl.store(Y + r3[:, None] * N + c[None, :], 0.,
                     (r3[:, None] < ROWS) & (c[None, :] < N))
    if WRITE_ONCE:
        for offset in range(tl.program_id(0) * 256, (TOTAL_ROWS - ROWS) * N, CTAS * 256):
            j = offset + tl.arange(0, 256)
            tl.store(Y + ROWS * N + j, 0., j < (TOTAL_ROWS - ROWS) * N)


def launch(a, p, ends, rows, transposed, geometry, *, write_once=False):
    n, k = p.shape if transposed else p.shape[::-1]
    if not (a.is_cuda and a.dtype == torch.bfloat16 and a.is_contiguous()
            and a.ndim == 2 and a.shape[1] == k and a.shape[0] >= rows > 0
            and rows % 32 == 0 and p.sources is None and ends.is_contiguous()
            and ends.dtype == torch.int32 and ends.numel() == p.experts
            and ends.device == a.device):
        raise ValueError('Require static row32 BF16 GPU inputs')
    if any(t.device != a.device or not t.is_contiguous() for t in
           (p.data, p.absmax, p.scale_state.absmax, p.scale_state.code, p.offsets, p.code)):
        raise ValueError('Packed metadata must be contiguous on the same GPU')
    bn, bk = geometry['bn'], geometry['bk']
    if (bn not in (64,128) or bk != 32 or geometry['warps'] != 4 or geometry['stages'] != 1):
        raise ValueError('Quad geometry not prepared')
    ctas = min(triton.cdiv(rows, 128) * triton.cdiv(n, bn),
               torch.cuda.get_device_properties(a.device).multi_processor_count * geometry['waves'])
    y = (a.new_empty if write_once else a.new_zeros)(a.shape[0], n)
    _mm[(ctas,)](a, p.data, p.absmax, p.scale_state.absmax, p.scale_state.code,
                 p.offsets, p.code, ends, y, n, k, p.experts, rows,
                 p.shape[0] * p.shape[1], transposed, p.blocksize,
                 p.scale_state.blocksize, bn, bk, ctas, a.shape[0], write_once,
                 num_warps=geometry['warps'], num_stages=geometry['stages'],
                 enable_fp_fusion=False)
    return y


def local_decode_plan(ends, group_tiles=4):
    """CPU metadata estimate: decode counts, not runtime speed or GPU evidence."""
    if torch.is_tensor(ends) and ends.is_cuda:raise ValueError('CPU metadata only')
    values=[int(x) for x in ends]
    if group_tiles not in (1,2,4) or not values or values!=sorted(values) or any(x<0 or x%32 for x in values):raise ValueError('Invalid row32 metadata')
    counts=[]
    for first in range(0,values[-1],32*group_tiles):
        ids=[sum(row>=end for end in values) for row in range(first,min(values[-1],first+32*group_tiles),32)]
        counts.append(len(set(ids)))
    return dict(row_tiles=values[-1]//32,local_weight_decodes=sum(counts),groups=len(counts))
