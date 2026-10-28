"""Experimental graph-local NF4 output retention, never automatically installed.

The packed expert custom autograd forward runs without grad recording. Retain
selected frozen-weight GEMM outputs only for that checkpoint invocation, then
rebuild the SAME expert autograd context during checkpoint recomputation. This
does not reuse language features across examples, updates, or checkpoints.

CPU tests are not GPU acceptance. Use default checkpoint determinism checks,
preserved RNG, complete gradient/optimizer checks and a measured memory budget.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import torch
from torch.utils.checkpoint import checkpoint

_ACTIVE = ContextVar('nf4_checkpoint_memo_v10', default=None)
ROLES = {'gate_proj', 'up_proj', 'down_proj'}


class _Scope:
    def __init__(self, owner):
        self.owner = owner
        self.entries = []
        self.bytes = 0
        self.cursor = 0
        self.replay = False

    def __del__(self):
        self.owner.live_bytes -= self.bytes

    @contextmanager
    def context(self, replay):
        self.replay = replay
        self.cursor = 0
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)


class MemoController:
    def __init__(self, roles=('gate_proj', 'up_proj'), max_bytes=2**30,
                 verify_recompute=False):
        self.configure(roles, max_bytes, verify_recompute)
        self.live_bytes = 0
        self.peak_bytes = 0
        self.counters = dict(forward_calls=0, replay_hits=0, replay_misses=0,
                             budget_skips=0)

    def configure(self, roles, max_bytes, verify_recompute=False):
        if getattr(self, 'live_bytes', 0):
            raise RuntimeError('Cannot change memo policy with a live graph')
        if not set(roles) <= ROLES or max_bytes < 0:
            raise ValueError('Invalid memo policy')
        self.roles = frozenset(roles)
        self.max_bytes = int(max_bytes)
        self.verify = bool(verify_recompute)

    def contexts(self):
        scope = _Scope(self)
        return scope.context(False), scope.context(True)

    def statistics(self):
        return dict(self.counters, live_bytes=self.live_bytes,
                    peak_bytes=self.peak_bytes, cap_bytes=self.max_bytes)

    def wrap(self, original, role_by_packed_id):
        def grouped_mm(a, packed, ends, rows, transposed):
            scope = _ACTIVE.get()
            if scope is None or scope.owner is not self or not transposed:
                return original(a, packed, ends, rows, transposed)
            if torch.is_grad_enabled():
                raise RuntimeError('Memo requires the expert custom-autograd forward')
            role = role_by_packed_id.get(id(packed))
            if role not in ROLES:
                raise RuntimeError('Unknown packed-weight identity')
            signature = (id(packed), role, tuple(a.shape), tuple(a.stride()),
                         a.dtype, a.device, int(rows), tuple(ends.shape),
                         ends.dtype, ends.device)
            if scope.replay:
                i = scope.cursor
                scope.cursor += 1
                if i >= len(scope.entries) or scope.entries[i][0] != signature:
                    raise RuntimeError('NF4 checkpoint replay call trace changed')
                _, cached, cached_version, ref_a, ref_ends = scope.entries[i]
                if cached is None:
                    self.counters['replay_misses'] += 1
                    return original(a, packed, ends, rows, transposed)
                if cached._version != cached_version:
                    raise RuntimeError('Retained output was mutated')
                if self.verify and (not torch.equal(a, ref_a) or
                                    not torch.equal(ends, ref_ends)):
                    raise RuntimeError('NF4 checkpoint replay values changed')
                self.counters['replay_hits'] += 1
                # Caller zeros padding in place: never expose the retained copy.
                return cached.clone()
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


def install_backends(controller, blocks, modules):
    """Explicitly patch isolated-worker backend modules; return undo closures."""
    roles = {id(p): name for block in blocks
             for name, p in block._packed_nf4.items() if name in ROLES}
    restores = []
    for module in modules:
        original = module.grouped_mm
        module.grouped_mm = controller.wrap(original, roles)
        restores.append((module, original))
    return restores


class LayerMemoPolicy:
    """Attach after existing retention hooks; preserve their long-input fallback."""
    def __init__(self, model, controller):
        self.controller = controller
        self.first = self.count = 0
        self.originals = [x._gradient_checkpointing_func for x in model.layers]
        self.handle = model.register_forward_pre_hook(self.select, with_kwargs=True)

    def select(self, model, args, kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None:
            x = kwargs.get('input_ids')
        if x is None and args:
            x = args[0]
        assert x is not None
        eligible = x.shape[1] <= 1024 and x.shape[0] * x.shape[1] <= 2048
        for i, (layer, original) in enumerate(zip(model.layers, self.originals)):
            selected = eligible and self.first <= i < self.first + self.count
            layer._gradient_checkpointing_func = (
                partial(checkpoint, use_reentrant=False, preserve_rng_state=True,
                        context_fn=self.controller.contexts)
                if selected and layer.gradient_checkpointing else original)

    def remove(self, model):
        self.handle.remove()
        for layer, original in zip(model.layers, self.originals):
            layer._gradient_checkpointing_func = original
