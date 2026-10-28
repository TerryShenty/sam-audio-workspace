"""Forward-only paired-nibble decode; fixed accepted row32/BK32 geometry.
Original BF16 scale/multiply/rounding and two dot accumulators are retained.
No persistent decoded-weight cache and no speed or memory-saving claim.
"""
import triton
import triton.language as tl
@triton.jit
def _bytepair_weight(DATA, SCALES, CODE4, expert, columns, block,
                     N:tl.constexpr,K:tl.constexpr,PER:tl.constexpr,
                     BLOCK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    even_k=block*BK+2*tl.arange(0,BK//2)
    index=expert*PER+columns[:,None]*K+even_k[None,:]
    mask=(columns[:,None]<N)&(even_k[None,:]<K)
    packed=tl.load(DATA+index//2,mask,other=0).to(tl.int32)
    scale=tl.load(SCALES+index//BLOCK,mask,other=0).to(tl.float32)
    high=(tl.load(CODE4+(packed>>4))*scale).to(tl.bfloat16)
    low=(tl.load(CODE4+(packed&15))*scale).to(tl.bfloat16)
    high=tl.where(mask,high,0.)
    low=tl.where(mask,low,0.)
    return tl.interleave(high,low).trans()

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
                tl.static_assert(TRANSPOSED and K % 2 == 0 and BLOCK % 2 == 0)
                w0 = _bytepair_weight(DATA, SCALES, CODE4, e0, c, block,
                                      N, K, PER, BLOCK, BN, BK)
                a0 = tl.load(A + r0[:, None] * K + kk[None, :],
                             (r0[:, None] < active) & (kk[None, :] < K), other=0)
                acc0 = tl.dot(a0, w0, acc0)
                if first + 32 < active:
                    a1 = tl.load(A + r1[:, None] * K + kk[None, :],
                                 (r1[:, None] < active) & (kk[None, :] < K), other=0)
                    if e0 == e1:
                        acc1 = tl.dot(a1, w0, acc1)
                    else:
                        w1 = _bytepair_weight(DATA, SCALES, CODE4, e1, c, block,
                                              N, K, PER, BLOCK, BN, BK)
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

