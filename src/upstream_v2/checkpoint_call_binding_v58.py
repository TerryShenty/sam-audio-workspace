"""Isolated actual HF checkpoint-call adapter; not a full-model source registry.

The installed HF layer passes partial(super().__call__, **kwargs), *args.
Only the single positional hidden tensor is entrusted to Torch saved-input
version checks. Keyword tensor/container metadata is checked separately.
The adapter retains a weak reference to the partial, never its activations.
Layer source getters still need production registry/config integration.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import weakref
import torch
from torch.utils.checkpoint import checkpoint
from replay_route_candidate_v58 import ReplayRouteMemo, BoundScope, tensor_stamp
from memo_transfer_candidate_v1 import _ReusableContext
from memo_transfer_runtime_candidate_v27 import PolicyMemoController


def keyword_stamp(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tensor_stamp(value), value.data_ptr(), value.requires_grad)
    if value is None or type(value) in (bool, int, float, str):
        return (type(value).__name__, value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(keyword_stamp(v) for v in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ('dict', tuple((k, keyword_stamp(value[k])) for k in sorted(value)))
    raise ValueError('Unsupported mutable checkpoint keyword input')


class CheckpointCallBinding:
    def __init__(self, function, args, layer, source_getter):
        if (not isinstance(function, partial) or function.args or
                getattr(function.func, '__self__', None) is not layer or
                not hasattr(function.func, '__func__')):
            raise ValueError('Require the actual HF layer-bound checkpoint partial')
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError('Only one direct saved hidden-state argument is supported')
        self.function = weakref.ref(function)
        self.layer_id = id(layer)
        self.callable_id = id(function.func.__func__)
        self.source_getter = source_getter

    def __call__(self):
        function = self.function()
        if (function is None or id(getattr(function.func, '__self__', None)) != self.layer_id
                or id(getattr(function.func, '__func__', None)) != self.callable_id
                or function.args):
            raise RuntimeError('Checkpoint callable was released or rebound')
        return (self.layer_id, self.callable_id, self.source_getter(),
                keyword_stamp(function.keywords))


class _LayerBoundScope(BoundScope):
    def __init__(self, owner, binding):
        self.get_binding = binding
        with owner.bind(binding):
            super().__init__(owner)

    @contextmanager
    def context(self, replay):
        with self.owner.bind(self.get_binding):
            with super().context(replay):
                yield


class LayerReplayMemo(ReplayRouteMemo, PolicyMemoController):
    """Preserves the existing stack's PolicyMemoController type/clone fallback."""
    def __init__(self, *, max_bytes, metadata_cap=16*2**20):
        self.bound = ContextVar('v58_layer_checkpoint_binding', default=None)
        super().__init__(self.read_binding, max_bytes=max_bytes, metadata_cap=metadata_cap)

    def assert_boundary(self):
        self.boundary()

    @contextmanager
    def bind(self, binding):
        token = self.bound.set(binding)
        try:
            yield
        finally:
            self.bound.reset(token)

    def read_binding(self):
        binding = self.bound.get()
        if binding is None:
            raise RuntimeError('Missing explicit per-call layer binding')
        return binding()

    def contexts(self):
        if self.enabled:
            raise RuntimeError('Enabled route replay requires bound checkpoint invocation')
        return PolicyMemoController.contexts(self)

    def contexts_for(self, binding):
        if not self.enabled or not self.consume_once or self.verify:
            raise RuntimeError('Bound route replay requires enabled consume-only policy')
        scope = _LayerBoundScope(self, binding)
        return _ReusableContext(scope, False), _ReusableContext(scope, True)


class LayerCheckpointInvoker:
    def __init__(self, layer, memo, source_getter):
        self.layer = weakref.ref(layer)
        self.memo, self.source_getter = memo, source_getter

    def __call__(self, function, *args):
        context_fn = self.memo.contexts
        if self.memo.enabled:
            binding = CheckpointCallBinding(function, args, self.layer(), self.source_getter)
            context_fn = partial(self.memo.contexts_for, binding)
        return checkpoint(function, *args, use_reentrant=False,
                          preserve_rng_state=True, context_fn=context_fn)
