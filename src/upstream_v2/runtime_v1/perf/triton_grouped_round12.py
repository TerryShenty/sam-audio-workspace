"""Research-only BF16 ragged expert GEMM. No automatic production installation."""
import torch
import triton
import triton.language as tl

BM=16
BN=64
BK=32
WARPS=4
STAGES=2

@triton.jit
def _mm(A,W,Ends,O,N:tl.constexpr,K:tl.constexpr,E:tl.constexpr,
        WS0:tl.constexpr,WS1:tl.constexpr,WS2:tl.constexpr,
        M,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);column=tl.program_id(1)
    ids=tl.arange(0,triton.next_power_of_2(E))
    ends=tl.load(Ends+ids,ids<E,other=2147483647)
    expert=tl.sum((tile*BM>=ends).to(tl.int32),0)
    r=tile*BM+tl.arange(0,BM)
    c=column*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+k
        aa=tl.load(A+r[:,None]*K+kk[None,:],(r[:,None]<M)&(kk[None,:]<K),other=0)
        ww=tl.load(W+expert*WS0+kk[:,None]*WS1+c[None,:]*WS2,
                   (kk[:,None]<K)&(c[None,:]<N),other=0)
        acc=tl.dot(aa,ww,acc)
    tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<M)&(c[None,:]<N))

def grouped_mm(a,w,ends,rows):
    assert a.is_contiguous() and a.dtype==w.dtype==torch.bfloat16
    assert rows%BM==0 and a.shape[1]==w.shape[1]
    output=a.new_empty(a.shape[0],w.shape[-1])
    _mm[(rows//BM,triton.cdiv(w.shape[-1],BN))](a,w,ends,output,
        w.shape[-1],a.shape[1],w.shape[0],*w.stride(),rows,BM,BN,BK,
        num_warps=WARPS,num_stages=STAGES)
    output[rows:]=0
    return output
