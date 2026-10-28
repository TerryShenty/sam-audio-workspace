"""Research only: dequantize immutable NF4 tiles inside grouped tensor-core GEMM."""
import torch
import triton
import triton.language as tl
BM=32
BN=128
BK=64
WARPS=4

@triton.jit
def _mm(A,DATA,ABS,ABS2,CODE2,OFFSETS,CODE4,Ends,O,
        N:tl.constexpr,K:tl.constexpr,E:tl.constexpr,ROWS,
        PER:tl.constexpr,TRANSPOSED:tl.constexpr,BLOCK:tl.constexpr,SCALE_BLOCK:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0);column=tl.program_id(1)
    ids=tl.arange(0,triton.next_power_of_2(E))
    ends=tl.load(Ends+ids,ids<E,other=2147483647)
    expert=tl.sum((tile*BM>=ends).to(tl.int32),0)
    r=tile*BM+tl.arange(0,BM);c=column*BN+tl.arange(0,BN)
    k=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+k
        aa=tl.load(A+r[:,None]*K+kk[None,:],(r[:,None]<ROWS)&(kk[None,:]<K),other=0)
        if TRANSPOSED:i=expert*PER+c[None,:]*K+kk[:,None]
        else:i=expert*PER+kk[:,None]*N+c[None,:]
        valid=(kk[:,None]<K)&(c[None,:]<N)
        q=tl.load(DATA+i//2,valid,other=0).to(tl.int32)
        nibble=tl.where(i%2==0,q>>4,q&15)
        si=i//BLOCK
        sc=tl.load(ABS+si,valid,other=0).to(tl.int32)
        scale=tl.load(CODE2+sc)*tl.load(ABS2+si//SCALE_BLOCK,valid,other=0)
        scale=scale+tl.load(OFFSETS+expert)
        weight=(tl.load(CODE4+nibble)*scale).to(tl.bfloat16)
        weight=tl.where(valid,weight,0.)
        acc=tl.dot(aa,weight,acc)
    tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<ROWS)&(c[None,:]<N))

def grouped_mm(a,p,ends,rows,transposed):
    assert a.is_contiguous() and a.dtype==torch.bfloat16 and p.sources is None
    n,k=p.shape if transposed else p.shape[::-1]
    assert a.shape[1]==k and rows%BM==0
    output=a.new_empty(a.shape[0],n)
    _mm[(rows//BM,triton.cdiv(n,BN))](a,p.data,p.absmax,p.scale_state.absmax,
        p.scale_state.code,p.offsets,p.code,ends,output,n,k,p.experts,rows,
        p.shape[0]*p.shape[1],transposed,p.blocksize,p.scale_state.blocksize,
        BM,BN,BK,num_warps=WARPS,num_stages=1,enable_fp_fusion=False)
    output[rows:]=0
    return output
