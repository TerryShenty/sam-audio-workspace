"""Isolated FP32 rowwise INT8 quantization, one Triton program per row."""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
@triton.jit
def quant_kernel(X,Q,S,K:tl.constexpr,RS:tl.constexpr,CS:tl.constexpr,B:tl.constexpr):
 r=tl.program_id(0);c=tl.arange(0,B)
 v=tl.load(X+r*RS+c*CS,c<K,other=0).to(tl.float32)
 s=tl.maximum(tl.max(tl.abs(v),0),1.e-8)/127.
 z=tl.minimum(tl.maximum(libdevice.nearbyint(v/s),-127.),127.).to(tl.int8)
 tl.store(Q+r*K+c,z,c<K);tl.store(S+r,s)
