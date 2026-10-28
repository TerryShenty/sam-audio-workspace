"""Isolated approximate frozen-base linear; original trainable LoRA stays unchanged."""
import types,torch

def quant_rows(x):
 from int8_rowquant_kernel import quant_kernel
 import triton
 assert x.ndim==2
 m,k=x.shape
 q=torch.empty((m,k),device=x.device,dtype=torch.int8)
 s=torch.empty((m,1),device=x.device,dtype=torch.float32)
 quant_kernel[(m,)](x,q,s,k,x.stride(0),x.stride(1),triton.next_power_of_2(k),num_warps=4)
 return q,s

def multiply(x,qw,sw):
 from int8_fused_dynamic_kernel import kernel
 import triton
 qx,sx=quant_rows(x)
 qx=qx.contiguous()
 m,k=qx.shape;n=qw.shape[1]
 y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 kernel[(triton.cdiv(m,64)*triton.cdiv(n,128),)](qx,qw,sx,sw,y,m,n,k,qw.stride(0),qw.stride(1),64,128,128,num_warps=4,num_stages=3)
 return y

class FrozenInt8(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,qf,sf,qb,sb,bias):
  ctx.save_for_backward(qb,sb);ctx.shape=x.shape;ctx.dtype=x.dtype
  out=multiply(x.reshape(-1,x.shape[-1]).to(torch.bfloat16),qf,sf)
  if bias is not None:out=out+bias
  return out.reshape(*x.shape[:-1],qf.shape[1])
 @staticmethod
 def backward(ctx,g):
  qb,sb=ctx.saved_tensors
  dx=multiply(g.reshape(-1,g.shape[-1]).to(torch.bfloat16),qb,sb)
  return dx.reshape(ctx.shape).to(ctx.dtype),None,None,None,None,None

def install(model):
 originals=[]
 for m in model.modules():
  if not isinstance(m,torch.nn.Linear) or m.weight.requires_grad or min(m.weight.shape)<1024:continue
  assert m.bias is None or not m.bias.requires_grad
  with torch.no_grad():
   qf,sf=quant_rows(m.weight);qb,sb=quant_rows(m.weight.t().contiguous())
  m.register_buffer('_int8_qf',qf.t(),persistent=False);m.register_buffer('_int8_sf',sf.t(),persistent=False)
  m.register_buffer('_int8_qb',qb.t(),persistent=False);m.register_buffer('_int8_sb',sb.t(),persistent=False)
  originals.append((m,m.forward))
  def forward(self,x):
   if x.numel()//x.shape[-1]<32:return torch.nn.functional.linear(x,self.weight,self.bias)
   return FrozenInt8.apply(x,self._int8_qf,self._int8_sf,self._int8_qb,self._int8_sb,self.bias)
  m.forward=types.MethodType(forward,m)
 assert originals
 return originals
