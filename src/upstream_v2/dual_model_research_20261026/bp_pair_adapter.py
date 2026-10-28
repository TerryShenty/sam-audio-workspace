"""Isolated gate/up NF4 forward fusion, never imported by an existing worker.

Two independent accumulators share each input tile, expert lookup and launch.
Weights remain separately packed; arithmetic order within each dot is unchanged.
No activation fusion, decode cache or cross-update tensor cache is introduced.
This component needs a future identity/runtime adapter and real GPU acceptance.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from bp_identity import geometry as select
from nf4_checkpoint_memo_v10 import _ACTIVE as _MEMO_ACTIVE

FLAG = 'nf4_gate_up_scaled_rowpair_v1'
_PAIR_ACTIVE = ContextVar('nf4_gate_up_scaled_rowpair_v1', default=None)


@triton.jit
def _weight(DATA, SCALES, CODE4, i, mask, BLOCK: tl.constexpr):
    q=tl.load(DATA+i//2,mask,other=0).to(tl.int32)
    nibble=tl.where(i%2==0,q>>4,q&15)
    scale=tl.load(SCALES+i//BLOCK,mask,other=0).to(tl.float32)
    w=(tl.load(CODE4+nibble)*scale).to(tl.bfloat16)
    return tl.where(mask,w,0.)

@triton.jit(do_not_specialize=['ROWS','TOTAL_ROWS'])
def _pair(A,GD,GS,GC4,UD,US,UC4,Ends,G,U,
          N:tl.constexpr,K:tl.constexpr,E:tl.constexpr,ROWS,
          PER:tl.constexpr,BLOCK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
          CTAS:tl.constexpr,TOTAL_ROWS,WRITE_ONCE:tl.constexpr):
    cols=tl.cdiv(N,BN)
    active=tl.load(Ends+E-1)
    ids=tl.arange(0,triton.next_power_of_2(E))
    ends=tl.load(Ends+ids,ids<E,other=2147483647)
    for linear in range(tl.program_id(0),tl.cdiv(ROWS,64)*cols,CTAS):
        first=(linear//cols)*64
        r0=first+tl.arange(0,32)
        r1=first+32+tl.arange(0,32)
        c=(linear%cols)*BN+tl.arange(0,BN)
        if first<active:
            e0=tl.sum((first>=ends).to(tl.int32),0)
            e1=tl.minimum(tl.sum((first+32>=ends).to(tl.int32),0),E-1)
            g0=tl.zeros((32,BN),tl.float32)
            g1=tl.zeros((32,BN),tl.float32)
            u0=tl.zeros((32,BN),tl.float32)
            u1=tl.zeros((32,BN),tl.float32)
            for block in range(tl.cdiv(K,BK)):
                kk=block*BK+tl.arange(0,BK)
                mask=(kk[:,None]<K)&(c[None,:]<N)
                index=c[None,:]*K+kk[:,None]
                a0=tl.load(A+r0[:,None]*K+kk[None,:],(r0[:,None]<active)&(kk[None,:]<K),other=0)
                wg0=_weight(GD,GS,GC4,e0*PER+index,mask,BLOCK)
                wu0=_weight(UD,US,UC4,e0*PER+index,mask,BLOCK)
                g0=tl.dot(a0,wg0,g0)
                u0=tl.dot(a0,wu0,u0)
                if first+32<active:
                    a1=tl.load(A+r1[:,None]*K+kk[None,:],(r1[:,None]<active)&(kk[None,:]<K),other=0)
                    if e0==e1:
                        g1=tl.dot(a1,wg0,g1)
                        u1=tl.dot(a1,wu0,u1)
                    else:
                        wg1=_weight(GD,GS,GC4,e1*PER+index,mask,BLOCK)
                        wu1=_weight(UD,US,UC4,e1*PER+index,mask,BLOCK)
                        g1=tl.dot(a1,wg1,g1)
                        u1=tl.dot(a1,wu1,u1)
            mask0=(r0[:,None]<active)&(c[None,:]<N)
            tl.store(G+r0[:,None]*N+c[None,:],g0,mask0)
            tl.store(U+r0[:,None]*N+c[None,:],u0,mask0)
            if first+32<active:
                mask1=(r1[:,None]<active)&(c[None,:]<N)
                tl.store(G+r1[:,None]*N+c[None,:],g1,mask1)
                tl.store(U+r1[:,None]*N+c[None,:],u1,mask1)
            elif WRITE_ONCE:
                mask1=(r1[:,None]<ROWS)&(c[None,:]<N)
                tl.store(G+r1[:,None]*N+c[None,:],0.,mask1)
                tl.store(U+r1[:,None]*N+c[None,:],0.,mask1)
        elif WRITE_ONCE:
            mask0=(r0[:,None]<ROWS)&(c[None,:]<N)
            mask1=(r1[:,None]<ROWS)&(c[None,:]<N)
            tl.store(G+r0[:,None]*N+c[None,:],0.,mask0)
            tl.store(U+r0[:,None]*N+c[None,:],0.,mask0)
            tl.store(G+r1[:,None]*N+c[None,:],0.,mask1)
            tl.store(U+r1[:,None]*N+c[None,:],0.,mask1)
    if WRITE_ONCE:
        for offset in range(tl.program_id(0)*256,(TOTAL_ROWS-ROWS)*N,CTAS*256):
            j=offset+tl.arange(0,256)
            tl.store(G+ROWS*N+j,0.,j<(TOTAL_ROWS-ROWS)*N)
            tl.store(U+ROWS*N+j,0.,j<(TOTAL_ROWS-ROWS)*N)

def enabled(config):
    value = config.get(FLAG, False)
    if type(value) is not bool:
        raise ValueError('Explicit boolean gate/up policy required')
    if value and (config.get('static') is not True or config.get('nf4_row_tile', 32) != 32):
        raise ValueError('Gate/up component requires static row32')
    if type(config.get('nf4_static_write_once', False)) is not bool:
        raise ValueError('Explicit boolean output initialization policy required')
    return value


def identity_suffix(config):
    return ('nf4_gate_up_scaled_rowpair_v1',) if enabled(config) else ()


def compatible(g, u):
    for name in ('shape', 'experts', 'blocksize'):
        if getattr(g, name) != getattr(u, name):
            raise ValueError('Gate/up packed metadata mismatch: ' + name)
    if (g is u or g.sources is not None or u.sources is not None or
            g.scale_state.blocksize != u.scale_state.blocksize):
        raise ValueError('Require distinct, compatible frozen packed projections')


def launch_pair(a,g,u,ends,rows,config,row_reuse):
    from nf4_row_scale_candidate_v44 import validate_table
    compatible(g,u)
    vg,vu=select(config,'gate_proj',True),select(config,'up_proj',True)
    if vg!=vu or (vg['bn'],vg['bk'],vg['warps'],vg['stages'],vg['waves'])!=(64,32,4,1,32):
        raise ValueError('Only fixed accepted pair geometry')
    gs=validate_table(g,row_reuse.tables[id(g)])
    us=validate_table(u,row_reuse.tables[id(u)])
    n,k=g.shape
    if not (a.is_cuda and a.dtype==torch.bfloat16 and a.is_contiguous() and a.ndim==2
            and a.shape[1]==k and a.shape[0]>=rows>0 and rows%32==0
            and ends.is_contiguous() and ends.dtype==torch.int32 and ends.numel()==g.experts
            and a.device==ends.device==g.data.device==u.data.device):
        raise ValueError('Invalid paired input')
    if gs.dtype!=torch.bfloat16 or us.dtype!=torch.bfloat16:raise ValueError('Fixed BF16 scale tables')
    once=config.get('nf4_static_write_once',False)
    alloc=a.new_empty if once else a.new_zeros
    gout,uout=alloc(a.shape[0],n),alloc(a.shape[0],n)
    ctas=min(triton.cdiv(rows,64)*triton.cdiv(n,64),torch.cuda.get_device_properties(a.device).multi_processor_count*32)
    _pair[(ctas,)](a,g.data,gs,g.code,u.data,us,u.code,ends,gout,uout,n,k,g.experts,rows,n*k,g.blocksize,64,32,ctas,a.shape[0],once,num_warps=4,num_stages=1,enable_fp_fusion=False)
    return gout,uout


@dataclass
class _Invocation:
    owner: object
    config: dict
    pending: object = None


class _Graph:
    def __init__(self, owner):
        self.owner = owner
        owner.live_graphs += 1
    def __del__(self):
        self.owner.live_graphs -= 1


class PairDispatch:
    """Install on an isolated static routing module BEFORE per-projection memo.

    Preserves gate/up/down memo call traces. On checkpoint replay, delegate to
    individual calls, avoiding speculative pair computation with partial hits.
    The surrounding future runtime must enforce B2/long-input dynamic fallback,
    compiler/family identity, restore descriptors and full GPU gates.
    """
    def __init__(self, routing, choose_config, blocks, *, memo=None, row_reuse):
        self.routing, self.module, self.choose = routing, routing.backend, choose_config
        self.memo = memo
        self.row_reuse = row_reuse
        self.bound_sources = {}
        self.bound_tables = None
        self.bound_roles = None
        self.live_graphs = self.active = 0
        self.closed = False
        self.counts = dict(pairs=0, paired_outputs=0, replay_fallback=0, geometry_fallback=0)
        if getattr(self.module, 'BM', None) != 32 or routing.PAD != 32:
            raise ValueError('Require the original static row32 routing/backend')
        self.pairs = {}
        for block in blocks:
            g, u = (block._packed_nf4[r] for r in ('gate_proj','up_proj'))
            compatible(g, u)
            if id(g) in self.pairs and (self.pairs[id(g)][0] is not g or self.pairs[id(g)][1] is not u):
                raise ValueError('Conflicting gate/up association')
            self.pairs[id(g)] = (g,u)
        if not self.pairs:
            raise ValueError('No known gate/up pairs')
        self.pair_parent = self.pairs
        self.pair_ids = {key: (id(value), id(value[0]), id(value[1])) for key,value in self.pairs.items()}
        self._boundary()
        self.original_mm, self.original_experts = self.module.grouped_mm, routing.Experts
        owner = self
        def grouped_mm(a, p, ends, rows, transposed):
            inv = _PAIR_ACTIVE.get()
            if inv is None or inv.owner is not owner or not transposed:
                return owner.original_mm(a,p,ends,rows,transposed)
            if inv.pending is not None:
                old_a, old_p, old_ends, old_rows, out = inv.pending
                if not (a is old_a and p is old_p and ends is old_ends and rows == old_rows):
                    raise RuntimeError('Paired projection call trace changed')
                inv.pending = None
                owner.counts['paired_outputs'] += 1
                return out
            owner.validate_sources(p)
            pair = owner.pairs.get(id(p))
            if pair is None or pair[0] is not p:
                return owner.original_mm(a,p,ends,rows,transposed)
            scope = _MEMO_ACTIVE.get()
            if scope is not None and scope.replay:
                owner.counts['replay_fallback'] += 1
                return owner.original_mm(a,p,ends,rows,transposed)
            cfg = inv.config
            if select(cfg,'gate_proj',True) != select(cfg,'up_proj',True):
                owner.counts['geometry_fallback'] += 1
                return owner.original_mm(a,p,ends,rows,transposed)
            g, u = launch_pair(a, *pair, ends, rows, cfg, owner.row_reuse)
            for role in ("gate_proj", "up_proj"):
                key=role+'_forward'
                owner.row_reuse.counts[key]+=1
                owner.row_reuse.scale_counts[key]+=1
                if 'forward_kernel' in cfg.get('nf4_row_reuse_waves',{}).get(role,{}):owner.row_reuse.wave_counts[key]+=1
            inv.pending = (a,pair[1],ends,rows,u)
            owner.counts['pairs'] += 1
            return g
        class PairedExperts(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                ctx.shared_input_graph = _Graph(owner)
                inv = _Invocation(owner, dict(owner.choose()))
                token = _PAIR_ACTIVE.set(inv)
                owner.active += 1
                try:
                    result = owner.original_experts.forward(ctx,*args)
                    if inv.pending is not None:
                        raise RuntimeError('Paired up output was not consumed')
                    return result
                finally:
                    inv.pending = None
                    owner.active -= 1
                    _PAIR_ACTIVE.reset(token)
            @staticmethod
            def backward(ctx, upstream):
                return owner.original_experts.backward(ctx,upstream)
        class DispatchExperts:
            @staticmethod
            def apply(*args):
                if owner.closed:
                    raise RuntimeError('Closed pair dispatch')
                return (PairedExperts if enabled(owner.choose()) else owner.original_experts).apply(*args)
        self.installed_mm, self.installed_experts = grouped_mm, DispatchExperts
        self.module.grouped_mm, routing.Experts = grouped_mm, DispatchExperts

    def configure_sources(self,cfg):
        self._boundary()
        self.bound_sources={}
        self.bound_tables=self.row_reuse.tables
        self.bound_roles=self.row_reuse.roles
        if enabled(cfg):
            from nf4_row_scale_candidate_v44 import validate_table
            for pair in self.pairs.values():
                for p in pair:
                    table=self.row_reuse.tables[id(p)]
                    validate_table(p,table)
                    self.bound_sources[id(p)]=(p,table,table.values,table.source,table.payload)

    def validate_sources(self,p):
        if self.row_reuse.tables is not self.bound_tables or self.row_reuse.roles is not self.bound_roles:
            raise RuntimeError('Pair scale/role parent replaced')
        if self.pairs is not self.pair_parent:raise RuntimeError('Pair mapping parent replaced')
        pair=self.pairs.get(id(p))
        if pair is not None:
            if self.pair_ids.get(id(p))!=(id(pair),id(pair[0]),id(pair[1])):raise RuntimeError('Pair selection replaced')
            for role,item in zip(('gate_proj','up_proj'),pair):
                if enabled(self.choose()):
                    bound=self.bound_sources.get(id(item));table=self.row_reuse.tables.get(id(item))
                    if (bound is None or bound[0] is not item or table is not bound[1]
                            or table.values is not bound[2] or table.source is not bound[3] or table.payload is not bound[4]):
                        raise RuntimeError('Pair scale selected binding replaced')
                mapping=self.row_reuse.roles.get(id(item))
                if mapping is None or mapping[0] is not item or mapping[1]!=role:raise RuntimeError('Pair role source changed')
        return pair

    def _boundary(self):
        if self.closed or self.live_graphs or self.active:
            raise RuntimeError('Release all graphs before changing pair dispatch')
        if self.memo is not None:
            if hasattr(self.memo,'assert_boundary'):
                self.memo.assert_boundary()
            elif self.memo.live_bytes:
                raise RuntimeError('Release memo graphs')

    def close(self):
        self._boundary()
        if (self.module.grouped_mm is not self.installed_mm or
                self.routing.Experts is not self.installed_experts):
            raise RuntimeError('Remove outer wrappers before pair dispatch')
        self.module.grouped_mm, self.routing.Experts = self.original_mm, self.original_experts
        self.closed = True
