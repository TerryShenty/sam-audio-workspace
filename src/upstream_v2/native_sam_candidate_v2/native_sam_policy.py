"""SAM-only functions with their own real module globals, safe for Dynamo guards.

Attention arithmetic copied from sealed SAM Attention.forward; only q/k/v
contiguity is optional. No cloned function globals or global F replacement.
"""
import types
import torch
import torch.nn.functional as F
from einops import rearrange

def attention_forward(m,x,cross_x=None,key_padding_mask=None,rope=None):
    xq=m.wq(x)
    if cross_x is not None:xk,xv=m.wk(cross_x),m.wv(cross_x)
    else:xk,xv=m.wk(x),m.wv(x)
    xk=m.reshape_heads(xk,m.n_kv_heads)
    xv=m.reshape_heads(xv,m.n_kv_heads)
    xq=m.reshape_heads(xq,m.n_heads)
    if m.use_qk_norm:xq=m.q_norm(xq);xk=m.k_norm(xk)
    if rope is not None:xq=rope(xq,bhle=True);xk=rope(xk,bhle=True)
    attn_mask=None
    if key_padding_mask is not None:attn_mask=key_padding_mask[:,None,None,:]
    if m._native_sam_contiguous:xq,xk,xv=xq.contiguous(),xk.contiguous(),xv.contiguous()
    output=F.scaled_dot_product_attention(xq,xk,xv,attn_mask=attn_mask)
    output=rearrange(output,'b h n d -> b n (h d)')
    return m.wo(output)

def rms_forward(m,x):
    if m._native_sam_rms and m.weight.dtype!=torch.float64:
        return F.rms_norm(x.float(),(x.shape[-1],),m.weight.float(),eps=m.eps).type_as(x)
    output=m._norm(x.float())
    return (output*m.weight).type_as(x)

class NativeSAMPolicy:
    def __init__(self,sam_root,attention_type,rms_type):
        self.contiguous=False;self.native_rms=False;self.handles=[]
        self.attention_paths=[];self.rms_paths=[];self.closed=False
        self.attentions=[];self.norms=[]
        try:
            for name,module in sam_root.named_modules():
                if isinstance(module,attention_type):
                    attr='_singleton_original' if hasattr(module,'_singleton_original') else 'forward'
                    original=getattr(module,attr)
                    if not isinstance(original,types.MethodType) or original.__func__ is not attention_type.forward:
                        raise ValueError('Install on original SAM Attention after singleton: '+name)
                    self._patch(module,'_native_sam_contiguous',False)
                    self._patch(module,attr,types.MethodType(attention_forward,module))
                    self.attentions.append(module);self.attention_paths.append(name)
                if isinstance(module,rms_type):
                    original=module.forward
                    if not isinstance(original,types.MethodType) or original.__func__ is not rms_type.forward:
                        raise ValueError('Unexpected SAM RMS wrapper: '+name)
                    self._patch(module,'_native_sam_rms',False)
                    self._patch(module,'forward',types.MethodType(rms_forward,module))
                    self.norms.append(module);self.rms_paths.append(name)
        except BaseException:self.close();raise
    def _patch(self,module,attr,replacement):
        self.handles.append((module,attr,attr in module.__dict__,module.__dict__.get(attr)))
        setattr(module,attr,replacement)
    def configure(self,mode):
        if self.closed:raise RuntimeError('Policy has been closed')
        if mode not in ('existing','contiguous','native_rms','both'):raise ValueError(mode)
        self.contiguous=mode in ('contiguous','both');self.native_rms=mode in ('native_rms','both')
        for m in self.attentions:m._native_sam_contiguous=self.contiguous
        for m in self.norms:m._native_sam_rms=self.native_rms
    def statistics(self):
        return dict(contiguous_sdpa=self.contiguous,native_fp32_rms=self.native_rms,
            attention_paths=list(self.attention_paths),rms_paths=list(self.rms_paths),
            global_torch_functions_changed=False,parameter_replacement=False,
            installed_paths_are_not_runtime_capture_proof=True,cloned_function_globals=False)
    def close(self):
        for module,attr,had,original in reversed(self.handles):
            if had:setattr(module,attr,original)
            else:delattr(module,attr)
        self.handles.clear();self.closed=True

def compile_native_sam_block(block, captures, backend_factory=None):
    """The native dynamic + GEMM-only autotune combination, no global config.

    Compile/capture and warmup belong outside throughput windows. Call before
    training; use a fresh compiled callable per policy mode for experiment
    isolation. The caller owns restoration of block.forward.
    """
    options = {'triton.cudagraphs': False, 'max_autotune_gemm': True}
    def backend(gm, inputs):
        if backend_factory is None:
            from torch._inductor import compile as inductor_compile
            compiled = inductor_compile(gm, inputs, options=options)
        else:
            compiled = backend_factory(gm, inputs, options=options)
        captures.append(dict(nodes=len(list(gm.graph.nodes)), backend_returned=True,
                             dynamic=True, options=dict(options)))
        return compiled
    return torch.compile(block.forward, backend=backend, fullgraph=True, dynamic=True)
