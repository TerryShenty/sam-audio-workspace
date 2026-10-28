"""Synthetic screening only. Tiling reference: Triton official matmul tutorial."""
import os,json,time
from pathlib import Path
import torch,triton
import triton.language as tl
@triton.jit
def kernel(A,B,SX,SW,Y,M,N:tl.constexpr,K:tl.constexpr,BKSTR:tl.constexpr,BNSTR:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 pid=tl.program_id(0);nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
 group=pid//(8*nn);first=group*8;gs=tl.minimum(nm-first,8)
 pm=first+(pid%(8*nn))%gs;pn=(pid%(8*nn))//gs
 rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.int32)
 for k in range(tl.cdiv(K,BK)):
  kk=k*BK+rk
  a=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),other=0)
  b=tl.load(B+kk[:,None]*BKSTR+rn[None,:]*BNSTR,(kk[:,None]<K)&(rn[None,:]<N),other=0)
  acc+=tl.dot(a,b)
 sx=tl.load(SX+rm,rm<M,other=0);sw=tl.load(SW+rn,rn<N,other=0)
 y=acc.to(tl.float32)*sx[:,None]*sw[None,:]
 tl.store(Y+rm[:,None]*N+rn[None,:],y,(rm[:,None]<M)&(rn[None,:]<N))
