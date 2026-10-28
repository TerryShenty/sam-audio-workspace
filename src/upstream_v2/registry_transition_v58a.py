"""Isolated follow-up to the accepted v58 registry fixture; not worker-installed.

Deduplicate only identical QuantState nodes while retaining every incoming
parent edge. Keep all three complete per-call reads and immutable validation.
Publish bindings only after a graph-free transition succeeds. State restoration
has a separate readiness bit, so configuration cannot repair a partial restore.
"""
import weakref
from layer_source_registry_v58 import LayerSource, GuardedLayerInvoker
from lora_policy_v21 import JointCheckpointPolicy


class SharedQuantSource(LayerSource):
    def __init__(self, *args, **kwargs):
        self._quant_seen = set()
        super().__init__(*args, **kwargs)
        del self._quant_seen

    def quant(self, state, depth=0):
        if state is None:
            return
        if depth > 1:
            raise ValueError('Only installed two-level quantization supported')
        if id(state) in self._quant_seen:
            return
        self._quant_seen.add(id(state))
        # The callers already registered weight.quant_state/module.quant_state
        # parent edges. This node's state2 edge is registered on its first visit.
        super().quant(state, depth)


class GenerationInvoker(GuardedLayerInvoker):
    def __init__(self, policy, layer, source, original):
        self.owner = weakref.ref(policy)
        self.generation = policy.epoch
        super().__init__(layer, policy.memo, source, original)

    def __call__(self, function, *args):
        policy = self.owner()
        if policy is None or policy.epoch != self.generation:
            raise RuntimeError('Stale checkpoint invoker')
        policy.require_step()
        # The generation gate precedes the inherited lazy fallback.
        return super().__call__(function, *args)


class TransitionRegistry(JointCheckpointPolicy):
    """Worker must call require_step before data/zero_grad, including update=False.

    extra_boundaries must include every actual runtime/backend/offload boundary.
    This isolated class does not assert that an arbitrary worker supplied them.
    The fixed candidate never enables a LoRA context (not a generic zero-byte
    context detector). All mutation callbacks propagate their exceptions.
    """
    source_type = SharedQuantSource

    def __init__(self, model, memo, extra_boundaries=()):
        if len(model.layers) != 48:
            raise ValueError('Require current 48-layer model')
        self.registry = None
        self.invokers = []
        self.epoch = 0
        self.binding_ready = False
        self.state_ready = False
        self.bootstrap_finished = False
        self.last_error = None
        self.extra_boundaries = tuple(extra_boundaries)
        super().__init__(model, memo)

    def initialization_complete(self):
        if self.bootstrap_finished:
            raise RuntimeError('Initialization cannot clear a failed restore')
        self.bootstrap_finished = True
        self.state_ready = True

    def require_step(self):
        if not self.binding_ready or not self.state_ready:
            raise RuntimeError('Incomplete binding or training-state transition')

    def boundary(self):
        self.memo.boundary()
        if self.linear.max_bytes or self.linear.live_bytes or self.active:
            raise RuntimeError('This candidate requires inactive zero-budget LoRA')
        for check in self.extra_boundaries:
            check()

    def _invalidate(self):
        self.binding_ready = False
        self.epoch += 1
        for layer, original in zip(self.model.layers, self.originals):
            layer._gradient_checkpointing_func = original
        self.registry = None
        self.invokers = []

    def _publish(self, tables, runtime_binding):
        sources = [self.source_type(self.model, i, tables, runtime_binding, self.epoch)
                   for i in range(48)]
        invokers = [GenerationInvoker(self, layer, source, original)
                    for layer, source, original in
                    zip(self.model.layers, sources, self.originals)]
        self.registry, self.invokers = sources, invokers
        self.binding_ready = True
        self.last_error = None

    def transition(self, validate, mutate, bindings):
        config = validate()  # Must be pure; malformed requests change nothing.
        self.boundary()     # All graph/lifetime checks precede all mutation.
        self._invalidate()  # Drop layer and policy owners before changing tables.
        try:
            mutate(config)
            tables, supplier = bindings()
            self._publish(tables, supplier)
        except Exception as error:
            self.last_error = str(error)  # Never retain exception/traceback.
            raise
        # Deliberately does not change state_ready.

    def configure(self, max_bytes):
        if type(max_bytes) is not int or max_bytes != 0:
            raise ValueError('This candidate requires zero LoRA cache')
        self.boundary()
        self._invalidate()
        super().configure(0)

    def rebuild(self, tables, runtime_binding):
        if not callable(runtime_binding):
            raise ValueError('Live binding supplier required')
        self.boundary()
        self._invalidate()
        try:
            self._publish(tables, runtime_binding)
        except Exception as error:
            self.last_error = str(error)
            raise

    def restore(self, validate, write_state):
        validate()  # Source/schema/config validation before any state change.
        self.boundary()
        self.bootstrap_finished = True
        self.state_ready = False
        try:
            write_state()  # Includes model, EMA, opt/sched, all RNG, final sync.
        except Exception as error:
            self.last_error = str(error)
            raise
        self.state_ready = True
        self.last_error = None

    def select(self, model, args, kwargs):
        self.require_step()  # Also precedes disabled delegation.
        if not self.memo.enabled:
            return super().select(model, args, kwargs)
        x = kwargs.get('inputs_embeds')
        if x is None:
            x = kwargs.get('input_ids')
        if x is None and args:
            x = args[0]
        if x is None:
            raise ValueError('Missing model input')
        eligible = x.shape[0] == 2 and x.shape[1] <= 1024
        self._activate(set())
        self.last_eligible, self.last_cached_layers = bool(eligible), []
        for i, (layer, original) in enumerate(zip(model.layers, self.originals)):
            layer.self_attn._qwen_lora_cache_variant = 'plain'
            selected = eligible and layer.gradient_checkpointing and self.first <= i < self.first+self.count
            layer._gradient_checkpointing_func = self.invokers[i] if selected else original

    def remove(self):
        self.boundary()
        self._invalidate()
        super().remove()
