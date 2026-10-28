"""Worker entry/transition wiring for the single v58 structural candidate.

Uninstalled: the worker must explicitly provide its pure config validator and
existing mutation/restore bodies. This module does not publish READY, validate
checkpoint provenance, or replace full-state/finite-training acceptance.
"""
from functools import wraps

from bp_live import LiveRuntimeBindings, POLICIES


class WorkerLifecycle:
    def __init__(self, registry, stack, compiler, runtimes, offload):
        if set(runtimes) != {p[0] for p in POLICIES}:
            raise ValueError('All four actual route runtimes are required')
        self.registry, self.stack, self.compiler = registry, stack, compiler
        self.runtimes, self.offload = dict(runtimes), offload
        self.in_step = self.in_transition = False
        # Merely constructing this adapter must not declare initialized model,
        # optimizer, scheduler, EMA, or RNG state ready.
        registry.extra_boundaries = tuple(registry.extra_boundaries) + (
            self.idle_boundary, stack._boundary,
            *(runtimes[p[0]].boundary for p in POLICIES), self.offload_boundary)

    def idle_boundary(self):
        if self.in_step:
            raise RuntimeError('Worker step has not finished')

    def offload_boundary(self):
        policy = self.offload.policy
        if policy is not None and policy.stats['live_host_bytes']:
            raise RuntimeError('Offload tensors remain live')
        # The chosen winner has no offload wrappers at all. This narrow contract
        # does not infer arbitrary context absence from zero retained bytes.
        if self.offload.patches or (policy is not None and policy.cap):
            raise RuntimeError('This fixed candidate requires offload disabled')

    def initialized(self):
        """Call once after the existing full state initialization has completed."""
        self.registry.initialization_complete()

    @staticmethod
    def validate_candidate(config):
        if type(config) is not dict:
            raise ValueError('Require a normalized configuration dictionary')
        for key in ('lora_cache_bytes', 'offload_bytes'):
            if type(config.get(key, 0)) is not int or config.get(key, 0) != 0:
                raise ValueError('This fixed candidate requires '+key+'=0')
        if config.get('nf4_gate_up_shared_input', False) is not False:
            raise ValueError('Shared-input fusion is outside this candidate')
        return config

    def configure(self, validate, mutate):
        """All prechecks precede any mutation; publish after the entire body.

        validate must include the worker's existing pure normalize/memory/policy
        checks. mutate must configure runtime/backend/table/memo/retention/native/
        compiler/forwards/train state, with its existing final synchronization.
        It must not call registry.configure or recursively configure this object.
        """
        if self.in_transition:
            raise RuntimeError('Nested configuration/restoration is unsupported')
        self.in_transition = True
        try:
            def checked():
                return self.validate_candidate(validate())
            def bindings():
                supplier = LiveRuntimeBindings(self.registry, self.stack,
                                               self.compiler, self.runtimes)
                return self.stack.row_reuse.tables, supplier
            self.registry.transition(checked, mutate, bindings)
        finally:
            self.in_transition = False

    def restore(self, validate, write_state):
        """Reuse accessors/tables/compiler for same-object complete restoration.

        The existing write body includes params/EMA/optimizer/scheduler/all RNG
        and final synchronization. A failed write leaves state_ready false;
        configure cannot clear that flag. No digest is added to each restore.
        """
        if self.in_transition:
            raise RuntimeError('Nested configuration/restoration is unsupported')
        self.in_transition = True
        try:
            self.registry.restore(validate, write_state)
        finally:
            self.in_transition = False

    def guard_step(self, operation):
        """Wrap the existing entire step, before input_copy/data/zero_grad.

        This includes update=False. Direct layer calls outside this worker are
        not covered by this entry gate; generated windows must call this wrapper.
        """
        @wraps(operation)
        def step(*args, **kwargs):
            if self.in_transition or self.in_step:
                raise RuntimeError('Worker is already in a state transition or step')
            self.registry.require_step()
            self.in_step = True
            try:
                return operation(*args, **kwargs)
            finally:
                self.in_step = False
        return step
