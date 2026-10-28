"""Uninstalled integration of v58a transitions and the per-layer live supplier.

This module still needs the affected DataMover/full-worker integration. It does
not alter the frozen v58a payload or the previously successful components.
"""
from registry_transition_v58a import SharedQuantSource, TransitionRegistry
from bp_live import LiveRuntimeBindings


class RuntimeLayerSource(SharedQuantSource):
    def __init__(self, model, index, tables, runtime_binding, epoch):
        local_binding = runtime_binding.for_layer(index, tables)
        super().__init__(model, index, tables, local_binding, epoch)


class RuntimeRegistry(TransitionRegistry):
    source_type = RuntimeLayerSource

    def __init__(self, *args, **kwargs):
        self.runtime_bindings = None
        super().__init__(*args, **kwargs)

    def _invalidate(self):
        supplier = self.runtime_bindings
        self.runtime_bindings = None
        super()._invalidate()
        if supplier is not None:
            supplier.close()

    def _publish(self, tables, runtime_binding):
        if (type(runtime_binding) is not LiveRuntimeBindings
                or runtime_binding.policy() is not self):
            raise ValueError('Require this registry\'s live runtime binding factory')
        try:
            runtime_binding.require_open()
            if tables is not runtime_binding.stack.row_reuse.tables:
                raise ValueError('Publication table parent differs from the live runtime')
            super()._publish(tables, runtime_binding)
        except Exception:
            runtime_binding.close()
            raise
        self.runtime_bindings = runtime_binding
