"""Isolated graph-local memo replay ownership transfer; never auto-installed.

Keep the original store clone: the routing caller mutates its output padding.
On first replay, remove the cached tensor from the memo before handing it to
that caller. A later replay recomputes instead of reusing the mutated tensor.
Keep the full original byte reservation until the checkpoint scope dies.
This changes no GEMM, precision, training data, or across-update caching.
CPU proof is not GPU acceptance or a speed measurement.
"""
from contextlib import contextmanager
import torch
import nf4_checkpoint_memo_v10 as base

_CONSUMED = object()


class _ReusableContext:
    """Checkpoint may enter recomputation again for a retained graph."""
    def __init__(self, scope, replay):
        self.scope, self.replay, self.entered = scope, replay, []

    def __enter__(self):
        if self.entered:
            raise RuntimeError('Recursive entry of the same checkpoint scope')
        manager = self.scope.context(self.replay)
        result = manager.__enter__()
        self.entered.append(manager)
        return result

    def __exit__(self, *exc):
        return self.entered.pop().__exit__(*exc)


class _Scope(base._Scope):
    @contextmanager
    def context(self, replay):
        self.owner.active_contexts += 1
        try:
            with super().context(replay):
                yield
        finally:
            self.owner.active_contexts -= 1


class TransferMemoController(base.MemoController):
    def __init__(self, roles=('gate_proj', 'up_proj'), max_bytes=2**30,
                 verify_recompute=False):
        self.active_contexts = 0
        super().__init__(roles, max_bytes, verify_recompute)
        self.counters.update(replay_transfers=0, replay_transfer_bytes=0,
                             repeated_replay_recomputes=0)

    def configure(self, roles, max_bytes, verify_recompute=False):
        if getattr(self, 'active_contexts', 0):
            raise RuntimeError('Cannot change memo policy inside an active context')
        return super().configure(roles, max_bytes, verify_recompute)

    def contexts(self):
        scope = _Scope(self)
        return _ReusableContext(scope, False), _ReusableContext(scope, True)

    def statistics(self):
        return dict(super().statistics(), replay_policy='consume_once_v1',
                    active_contexts=self.active_contexts,
                    reservation_held_until_scope_release=True)

    def wrap(self, original, role_by_packed_id):
        def grouped_mm(a, packed, ends, rows, transposed):
            scope = base._ACTIVE.get()
            if scope is None or scope.owner is not self or not transposed:
                return original(a, packed, ends, rows, transposed)
            if torch.is_grad_enabled():
                raise RuntimeError('Memo requires the expert custom-autograd forward')
            role = role_by_packed_id.get(id(packed))
            if role not in base.ROLES:
                raise RuntimeError('Unknown packed-weight identity')
            signature = (id(packed), role, tuple(a.shape), tuple(a.stride()),
                         a.dtype, a.device, int(rows), tuple(ends.shape),
                         ends.dtype, ends.device)
            if scope.replay:
                i = scope.cursor
                scope.cursor += 1
                if i >= len(scope.entries) or scope.entries[i][0] != signature:
                    raise RuntimeError('NF4 checkpoint replay call trace changed')
                _, cached, version, ref_a, ref_ends = scope.entries[i]
                if self.verify and ref_a is not None and (
                        not torch.equal(a, ref_a) or not torch.equal(ends, ref_ends)):
                    raise RuntimeError('NF4 checkpoint replay values changed')
                if cached is None or cached is _CONSUMED:
                    self.counters['replay_misses'] += 1
                    if cached is _CONSUMED:
                        self.counters['repeated_replay_recomputes'] += 1
                    return original(a, packed, ends, rows, transposed)
                if cached._version != version:
                    raise RuntimeError('Retained output was mutated')
                # Remove ownership BEFORE caller padding writes; never retain
                # a writable alias for later replay. Do not release reservation.
                scope.entries[i] = (signature, _CONSUMED, version, ref_a, ref_ends)
                self.counters['replay_hits'] += 1
                self.counters['replay_transfers'] += 1
                self.counters['replay_transfer_bytes'] += cached.numel() * cached.element_size()
                return cached
            self.counters['forward_calls'] += 1
            out = original(a, packed, ends, rows, transposed)
            size = out.numel() * out.element_size()
            if self.verify:
                size += a.numel() * a.element_size() + ends.numel() * ends.element_size()
            cached = ref_a = ref_ends = version = None
            if role in self.roles and self.live_bytes + size <= self.max_bytes:
                cached = out.detach().clone()
                version = cached._version
                if self.verify:
                    ref_a, ref_ends = a.detach().clone(), ends.detach().clone()
                scope.bytes += size
                self.live_bytes += size
                self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            elif role in self.roles:
                self.counters['budget_skips'] += 1
            scope.entries.append((signature, cached, version, ref_a, ref_ends))
            return out
        return grouped_mm
