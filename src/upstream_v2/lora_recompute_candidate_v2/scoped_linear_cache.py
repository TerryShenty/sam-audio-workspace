"""CPU-preparation candidate: bounded graph-local selective LoRA matmul saves.

Never auto-installed. Only wraps explicit LoRA A/B nn.Linear leaves, so SAC
does not observe the changing internal trace of the separate NF4 memo backend.
Requires the installed PyTorch >=2.13 forward-only policy/op_output API.
GPU speed and production use are unverified. Cache bookkeeping and the two
small trainable LoRA leaves deliberately execute eagerly to prevent Dynamo
from specializing on graph-local replay lists and cursors.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import types
import weakref
import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

_ACTIVE = ContextVar('lora_recompute_candidate_v2_scope', default=None)
OPS = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}


class _Scope:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []
        self.cursor = 0
        self.replay = False
        self.bytes = 0

    @contextmanager
    def context(self, replay):
        self.replay, self.cursor = replay, 0
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def __del__(self):
        self.owner.live_bytes -= self.bytes


class ScopedLinearCache:
    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError('Negative budget')
        self.live_bytes = self.peak_bytes = 0
        self.saved_matmuls = self.budget_skips = self.replayed_calls = 0
        self.forward_calls = self.fallback_calls = 0
        self.installed = []

    def configure(self, max_bytes):
        if self.live_bytes:
            raise RuntimeError('Cannot change policy with a live checkpoint graph')
        if max_bytes < 0:
            raise ValueError('Negative budget')
        self.max_bytes = int(max_bytes)

    def contexts(self):
        scope = _Scope(self)
        return scope.context(False), scope.context(True)

    def combined_contexts(self, other_context_fn):
        """Compose context managers without replacing any saved-tensor hooks."""
        own = self.contexts()
        other = other_context_fn()
        @contextmanager
        def combined(i):
            with ExitStack() as stack:
                stack.enter_context(other[i])
                stack.enter_context(own[i])
                yield
        return combined(0), combined(1)

    def _modes(self, scope):
        scope_ref = weakref.ref(scope)
        def policy(ctx, op, *args, **kwargs):
            if op not in OPS:
                return CheckpointPolicy.MUST_RECOMPUTE
            if not hasattr(ctx, 'op_output') or ctx.is_recompute:
                raise RuntimeError('This candidate requires forward-only SAC op_output API')
            out = ctx.op_output
            if not torch.is_tensor(out) or out.layout != torch.strided:
                return CheckpointPolicy.MUST_RECOMPUTE
            size = out.numel() * out.element_size()
            # These ATen matmuls should own a dense output; reject storage views.
            if out.untyped_storage().nbytes() != size or out.storage_offset() != 0:
                return CheckpointPolicy.MUST_RECOMPUTE
            active = scope_ref()
            if active is None:
                raise RuntimeError('Checkpoint scope expired')
            if self.live_bytes + size > self.max_bytes:
                self.budget_skips += 1
                return CheckpointPolicy.MUST_RECOMPUTE
            self.live_bytes += size
            active.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            self.saved_matmuls += 1
            return CheckpointPolicy.MUST_SAVE
        return create_selective_checkpoint_contexts(policy, allow_cache_entry_mutation=False)

    def install(self, model):
        if self.installed:
            raise RuntimeError('Already installed')
        owner = self
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not ({'lora_A', 'lora_B'} & set(name.split('.'))):
                continue
            original = module.forward
            @torch.compiler.disable
            def forward(mod, x, _original=original):
                scope = _ACTIVE.get()
                if scope is None or scope.owner is not owner:
                    owner.fallback_calls += 1
                    return _original(x)
                signature = (id(mod), tuple(x.shape), tuple(x.stride()), x.dtype,
                             x.device, mod.weight._version,
                             None if mod.bias is None else mod.bias._version)
                if not scope.replay:
                    modes = owner._modes(scope)
                    scope.calls.append((signature, modes))
                    owner.forward_calls += 1
                    mode = modes[0]
                else:
                    i = scope.cursor
                    scope.cursor += 1
                    if i >= len(scope.calls) or scope.calls[i][0] != signature:
                        raise RuntimeError('LoRA checkpoint call trace or parameter version changed')
                    mode = scope.calls[i][1][1]
                    owner.replayed_calls += 1
                with mode:
                    return _original(x)
            module.forward = types.MethodType(forward, module)
            self.installed.append((name, module, original))
        return [name for name, _, _ in self.installed]

    def remove(self):
        if self.live_bytes:
            raise RuntimeError('Cannot remove policy with a live checkpoint graph')
        for _, module, original in self.installed:
            module.forward = original
        self.installed.clear()

    def statistics(self):
        return dict(cap_bytes=self.max_bytes, live_bytes=self.live_bytes,
                    peak_bytes=self.peak_bytes, saved_matmuls=self.saved_matmuls,
                    budget_skips=self.budget_skips, replayed_calls=self.replayed_calls,
                    forward_calls=self.forward_calls, fallback_calls=self.fallback_calls,
                    installed_paths=[n for n, _, _ in self.installed],
                    across_update_cache=False, gpu_tested=False)
