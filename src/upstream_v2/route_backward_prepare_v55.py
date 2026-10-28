"""Isolated route backward preparation fusion; no installation or tensor cache."""
import weakref
import torch
import triton
import triton.language as tl

@triton.jit
def _prepare(G, Y, W, TOKEN, POS, DST, WEIGHTED, GW,
             D:tl.constexpr, BLOCK:tl.constexpr):
    route=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    token=tl.load(TOKEN+route); pos=tl.load(POS+route); dst=tl.load(DST+route)
    g=tl.load(G+token*D+col,col<D,0).to(tl.float32)
    y=tl.load(Y+dst*D+col,col<D,0).to(tl.float32)
    weight=tl.load(W+pos).to(tl.float32)
    # Materialized BF16 product rounding is retained; tree reduction may differ.
    product=(g*y).to(GW.dtype.element_ty).to(tl.float32)
    dw=tl.sum(product,0)
    tl.store(GW+pos,dw)
    tl.store(WEIGHTED+dst*D+col,g*weight,col<D)

def eligible(upstream,y,weights,token,positions,dst):
    if upstream.ndim!=2 or y.ndim!=2 or weights.ndim!=2:return False
    n,d=upstream.shape
    if d<1 or d>8192 or y.shape[1]!=d or weights.shape[0]!=n:return False
    if not 1<=weights.shape[1]<=16:return False
    values=(upstream,y,weights)
    if not all(t.is_cuda and t.dtype==torch.bfloat16 and t.is_contiguous() and t.device==y.device for t in values):return False
    routes=weights.numel()
    return all(t.ndim==1 and t.numel()==routes and t.is_cuda and t.is_contiguous() and t.device==y.device and t.dtype in (torch.int32,torch.int64) for t in (token,positions,dst))

class RouteBackwardPolicy:
    def __init__(self):
        self.mode='disabled';self.warps=4;self._live=weakref.WeakSet()
        self.counts=dict(calls=0,fallbacks=0,kernel_launches=0,route_rows=0,elided_expanded_tensor_bytes=0)
    def boundary(self):
        if self._live:raise RuntimeError('Cannot reconfigure route backward prep with live graphs')
    def configure(self,mode='disabled',warps=4):
        self.boundary()
        if mode not in ('disabled','fused') or type(warps) is not int or warps not in (4,8):raise ValueError('Unsupported backward prep policy')
        self.mode,self.warps=mode,warps
    def attach(self,ctx):
        if self.mode=='disabled':return None
        ctx.route_backward_warps=self.warps;self._live.add(ctx);return self
    def prepare(self,ctx,upstream,y,weights,token,positions,dst):
        if not eligible(upstream,y,weights,token,positions,dst):
            self.counts['fallbacks']+=1;return None
        # Padding and unrouted output rows must remain zero for later GEMMs.
        weighted=torch.zeros_like(y);gw=torch.empty_like(weights)
        routes=weights.numel();d=y.shape[1]
        if routes:
            _prepare[(routes,)](upstream,y,weights,token,positions,dst,weighted,gw,d,triton.next_power_of_2(d),num_warps=ctx.route_backward_warps,enable_fp_fusion=False)
            self.counts['kernel_launches']+=1
        self.counts['calls']+=1;self.counts['route_rows']+=routes
        # Four route-by-width intermediates: dy, gathered y, product, weighted dy.
        self.counts['elided_expanded_tensor_bytes']+=4*routes*d*y.element_size()
        return weighted,gw
    def release(self,ctx):self._live.discard(ctx)
    def statistics(self):
        return dict(self.counts,mode=self.mode,warps=self.warps,live_graphs=len(self._live),gpu_validated=False,structural_bytes_are_not_peak_reduction=True)
