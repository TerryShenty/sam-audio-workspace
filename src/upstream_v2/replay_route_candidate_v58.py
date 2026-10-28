"""Isolated atomic checkpoint replay bypass. Disabled unless explicitly enabled.

No worker imports this component. The source_binding callback must cover ALL
mutable tensors/configuration in the enclosing checkpoint function, not only
expert weights. It returns immutable host metadata (identity/version), never
activation tensors. Ordinary checkpoint replay with preserved RNG is required;
this contract is not value equality for arbitrary equal-shaped inputs.
"""
from contextlib import contextmanager
import torch
import nf4_checkpoint_memo_v10 as base
from memo_transfer_candidate_v1 import TransferMemoController, _Scope, _ReusableContext, _CONSUMED

ROLES = ('gate_proj', 'up_proj', 'down_proj')

def tensor_stamp(t):
    return (id(t), t._version, tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

class BoundScope(_Scope):
    def __init__(self, owner):
        super().__init__(owner)
        self.routes = {}
        self.route_bytes = 0
        self.counted = False
        self.binding = owner.source_stamp()
        owner.outstanding_scopes += 1
        self.counted = True

    @contextmanager
    def context(self, replay):
        if self.binding != self.owner.source_stamp():
            raise RuntimeError('Checkpoint source/configuration changed')
        with super().context(replay):
            yield

    def __del__(self):
        super().__del__()
        self.owner.metadata_reserved_bytes -= self.route_bytes
        if self.counted:
            self.owner.outstanding_scopes -= 1

class ReplayRouteMemo(TransferMemoController):
    def __init__(self, source_binding, *, max_bytes, metadata_cap=16*2**20):
        if not callable(source_binding):
            raise ValueError('An explicit full-checkpoint source binding is required')
        self.source_binding = source_binding
        self.enabled = False
        self.outstanding_scopes = 0
        self.metadata_reserved_bytes = self.metadata_peak_bytes = 0
        self.metadata_cap = metadata_cap
        super().__init__(ROLES, max_bytes, verify_recompute=False)
        self.counters.update(route_captures=0, route_hits=0, route_misses=0,
            metadata_budget_skips=0, bypassed_input_calls=0, bypassed_pointwise_calls=0)

    def boundary(self):
        if self.outstanding_scopes or getattr(self, 'active_contexts', 0) or getattr(self, 'live_bytes', 0) or self.metadata_reserved_bytes:
            raise RuntimeError('Release every checkpoint scope before policy changes')

    def configure(self, roles, max_bytes, verify_recompute=False):
        self.boundary()
        return super().configure(roles, max_bytes, verify_recompute)

    def set_enabled(self, enabled):
        self.boundary()
        if type(enabled) is not bool:
            raise ValueError('enabled must be bool')
        self.enabled = enabled

    def source_stamp(self):
        value = self.source_binding()
        def check(v):
            if isinstance(v, tuple):
                return all(check(x) for x in v)
            return v is None or isinstance(v, (bool,int,float,str,torch.dtype,torch.device))
        if not isinstance(value, tuple) or not check(value):
            raise ValueError('Source binding must contain immutable metadata, never tensors')
        return value

    def contexts(self):
        scope = BoundScope(self)
        return _ReusableContext(scope, False), _ReusableContext(scope, True)

    def scope(self):
        scope = base._ACTIVE.get()
        return scope if scope is not None and scope.owner is self else None

    @staticmethod
    def route_signature(x, selected, packed, rows, policies):
        # Shapes are only a call-trace guard INSIDE the bound checkpoint scope.
        return (tuple(x.shape), tuple(x.stride()), x.dtype, x.device,
            tuple(selected.shape), selected.dtype, selected.device, rows,
            tuple((id(packed[r]), tuple(packed[r].shape)) for r in ROLES), policies)

    def capture(self, x, selected, packed, rows, policies, metadata):
        scope = self.scope()
        if not self.enabled or scope is None or scope.replay or self.verify:
            return
        i = len(scope.entries)-3
        if i < 0:
            raise RuntimeError('Three expert calls must precede route capture')
        entries = scope.entries[i:i+3]
        if any(entry[1] is None or entry[1] is _CONSUMED for entry in entries):
            return
        if tuple(entry[0][1] for entry in entries) != ROLES:
            raise RuntimeError('Route capture expert trace changed')
        if i in scope.routes:
            raise RuntimeError('Repeated route capture at the same trace index')
        size = sum(t.numel()*t.element_size() for t in metadata)
        if self.metadata_reserved_bytes+size > self.metadata_cap:
            self.counters['metadata_budget_skips'] += 1
            return
        if any(t.dtype not in (torch.int32,torch.int64) or t.requires_grad for t in metadata):
            raise RuntimeError('Only integer route metadata may be retained')
        scope.routes[i] = (self.route_signature(x,selected,packed,rows,policies),
            metadata, tuple(tensor_stamp(t) for t in metadata))
        scope.route_bytes += size
        self.metadata_reserved_bytes += size
        self.metadata_peak_bytes = max(self.metadata_peak_bytes,self.metadata_reserved_bytes)
        self.counters['route_captures'] += 1

    def take(self, x, selected, packed, rows, policies, pointwise):
        scope = self.scope()
        if not self.enabled or scope is None or not scope.replay or self.verify:
            return None
        i = scope.cursor
        record = scope.routes.get(i)
        if record is None:
            self.counters['route_misses'] += 1
            return None
        signature,metadata,stamps = record
        if signature != self.route_signature(x,selected,packed,rows,policies):
            raise RuntimeError('Route replay trace/configuration changed')
        if stamps != tuple(tensor_stamp(t) for t in metadata):
            raise RuntimeError('Retained route metadata was mutated')
        entries = scope.entries[i:i+3]
        if len(entries) != 3:
            raise RuntimeError('Route replay expert trace is incomplete')
        if any(e[1] is None or e[1] is _CONSUMED for e in entries):
            self.counters['route_misses'] += 1
            return None  # No cursor/entry mutation: the complete old path still owns it.
        ends = metadata[-1]
        d = x.shape[1]
        f = packed['gate_proj'].shape[0]
        for role,width,entry in zip(ROLES,(d,d,f),entries):
            expected = (id(packed[role]),role,(rows+16,width),(width,1),
                x.dtype,x.device,rows,tuple(ends.shape),ends.dtype,ends.device)
            if entry[0] != expected:
                raise RuntimeError('Atomic memo call signature changed')
            if entry[1]._version != entry[2]:
                raise RuntimeError('Retained output was mutated')
        g,u,y = (entry[1] for entry in entries)
        # Must establish the same saved-tensor branch before transferring anything.
        if not pointwise.select(g,u):
            return None
        for offset,entry in enumerate(entries):
            sig,cached,version,ref_a,ref_ends = entry
            scope.entries[i+offset] = (sig,_CONSUMED,version,ref_a,ref_ends)
        del scope.routes[i]  # Remove owning cache references before returning aliases.
        scope.cursor += 3
        # The initial store clone precedes the original caller's tail zeroing.
        # Reproduce that write only AFTER removing all owning cache references.
        for out in (g,u,y):
            out[rows:] = 0
        self.counters['replay_hits'] += 3
        self.counters['replay_transfers'] += 3
        self.counters['replay_transfer_bytes'] += sum(t.numel()*t.element_size() for t in (g,u,y))
        self.counters['route_hits'] += 1
        self.counters['bypassed_input_calls'] += 1
        self.counters['bypassed_pointwise_calls'] += 1
        return (*metadata,g,u,y)

    def statistics(self):
        return dict(super().statistics(),enabled=self.enabled,outstanding_scopes=self.outstanding_scopes,
            metadata_reserved_bytes=self.metadata_reserved_bytes,metadata_peak_bytes=self.metadata_peak_bytes,
            metadata_cap_bytes=self.metadata_cap,gpu_validated=False,worker_installed=False)
