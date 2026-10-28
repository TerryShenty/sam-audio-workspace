"""Isolated forward route metadata/gather fusion. Not installed in any worker."""
import torch
import triton
import triton.language as tl

@triton.jit
def _metadata(SELECTED, ORDER, PADDED, ENDS, STARTS, TOKEN, POS, DST, INDICES, SLOTS,
              N:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    route=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    ok=route<N*K
    ordered=tl.load(ORDER+route,ok,0)
    token=ordered%N;slot=ordered//N;pos=token*K+slot
    expert=tl.load(SELECTED+pos,ok,0)
    end=tl.load(ENDS+expert,ok,0);padded=tl.load(PADDED+expert,ok,0)
    start=tl.load(STARTS+expert,ok,0)
    dst=end-padded+route-start
    # Top-k indices are unique per token. Rank is the unchanged ascending expert order.
    rank=tl.full((BLOCK,),0,tl.int32)
    for j in range(K):
        other=tl.load(SELECTED+token*K+j,ok,0)
        rank+=other<expert
    tl.store(TOKEN+route,token,ok);tl.store(POS+route,pos,ok);tl.store(DST+route,dst,ok)
    tl.store(INDICES+token*K+rank,dst,ok);tl.store(SLOTS+token*K+rank,slot,ok)

@triton.jit
def _gather(X, ORDER, COUNTS, PADDED, ENDS, STARTS, OUT,
            N:tl.constexpr, D:tl.constexpr, E:tl.constexpr, ROWS:tl.constexpr,
            SEARCH:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    lo=0;hi=E
    # Upper bound handles repeated cumulative ends for empty experts.
    for j in range(SEARCH):
        mid=(lo+hi)//2
        end=tl.load(ENDS+mid,mid<E,2147483647)
        right=(mid<E)&(row>=end)
        lo=tl.where(right,mid+1,lo);hi=tl.where(right,hi,mid)
    expert=lo;live=expert<E
    end=tl.load(ENDS+expert,live,0);pad=tl.load(PADDED+expert,live,0)
    count=tl.load(COUNTS+expert,live,0);start=tl.load(STARTS+expert,live,0)
    local=row-(end-pad);valid=live&(local>=0)&(local<count)
    ordered=tl.load(ORDER+start+local,valid,0);token=ordered%N
    value=tl.load(X+token*D+col,valid&(col<D),0)
    tl.store(OUT+row*D+col,value,col<D)

def eligible(x,selected,order,counts,padded,ends,starts,rows):
    if x.ndim!=2 or selected.ndim!=2:return False
    n,d=x.shape;k=selected.shape[1];e=counts.numel()
    if n<1 or not 1<=d<=8192 or selected.shape[0]!=n or not 1<=k<=min(16,e) or not 1<=e<=256:return False
    if type(rows) is not int or rows<n*k or rows>=2**31-16 or n*k>=2**31:return False
    if not (x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()):return False
    shapes=(selected.shape,(n*k,),(e,),(e,),(e,),(e,))
    return all(t.is_cuda and t.device==x.device and t.is_contiguous() and t.dtype in (torch.int32,torch.int64) and tuple(t.shape)==tuple(shape) for t,shape in zip((selected,order,counts,padded,ends,starts),shapes))

class RouteInputPolicy:
    def __init__(self):
        self.mode='disabled';self.block=1024
        self.counts=dict(calls=0,fallbacks=0,kernel_launches=0,route_rows=0,padded_rows_written=0,elided_gather_tensor_bytes=0)
    def configure(self,mode='disabled',block=1024):
        if mode not in ('disabled','fused') or type(block) is not int or block not in (256,512,1024):raise ValueError('Unsupported route input preparation policy')
        self.mode,self.block=mode,block
    def prepare(self,x,selected,order,counts,padded,ends,starts,rows):
        if self.mode=='disabled':return None
        if not eligible(x,selected,order,counts,padded,ends,starts,rows):
            self.counts['fallbacks']+=1;return None
        n,d=x.shape;k=selected.shape[1];e=counts.numel()
        token=torch.empty_like(order);pos=torch.empty_like(order);dst=torch.empty_like(order)
        indices=torch.empty_like(selected);slots=torch.empty_like(selected);routed=x.new_empty((rows+16,d))
        _metadata[(triton.cdiv(n*k,128),)](selected,order,padded,ends,starts,token,pos,dst,indices,slots,n,k,128,num_warps=4)
        _gather[(rows+16,triton.cdiv(d,self.block))](x,order,counts,padded,ends,starts,routed,n,d,e,rows+16,e.bit_length(),self.block,num_warps=4)
        self.counts['calls']+=1;self.counts['kernel_launches']+=2;self.counts['route_rows']+=n*k
        self.counts['padded_rows_written']+=rows+16;self.counts['elided_gather_tensor_bytes']+=n*k*d*x.element_size()
        return routed,token,pos,dst,indices,slots
    def statistics(self):
        return dict(self.counts,mode=self.mode,block=self.block,live_graphs=0,tensor_cache=False,gpu_validated=False,structural_bytes_are_not_peak_reduction=True)
