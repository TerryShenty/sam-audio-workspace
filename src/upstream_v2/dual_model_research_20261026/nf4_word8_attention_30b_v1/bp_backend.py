"""Isolated shared-input stack; existing v26/v27 workers never import it.

Paired forward is installed after write-once and before diagnostics/memo.
Checkpoint replay and long/B1/B3 retain individual/dynamic paths. Diagnostic
ranges describe logical calls, not independently timed gate/up kernels.
"""
from directional_backend_v9 import install as directional
from pipeline_launch_candidate_v12 import install as pipeline
from row_tile_runtime_v23 import install as row_install, dispatch
from projection_backend_candidate_v25 import install_backend, packed_roles
from static_write_once_candidate_v1 import install as write_once
from projection_diagnostic_candidate_v1 import ProjectionProbe
from nf4_checkpoint_memo_v10 import install_backends
from bp_pair_adapter import FLAG
from bp_identity import normalize, stamp_continuation, validate_resume_config
from bp_dispatch import RowReuse
from memo_transfer_runtime_candidate_v27 import PolicyMemoController
from bp_pair_adapter import PairDispatch, compatible
from contextlib import contextmanager


class PairAwareProbe(ProjectionProbe):
    """Logical call ranges; a fused gate range contains BOTH projection kernels."""
    @contextmanager
    def capture(self, *, max_calls=4096):
        before = dict(self.pair.counts)
        with super().capture(max_calls=max_calls) as report:
            report.update(shared_input_policy=bool(self.choose().get(FLAG)),
                          role_ranges_are_logical_calls=True,
                          paired_gate_range_includes_up=True,
                          up_return_range_is_not_up_kernel_time=True,
                          independent_six_role_device_costs=False)
            try:
                yield report
            finally:
                report['shared_input_counts'] = {k: v-before[k] for k, v in self.pair.counts.items()}



class BackendStack:
    def __init__(self, dynamic, static, routing, alias, blocks, memo, config, *, diagnostics=False):
        self.current = normalize(config)
        blocks = list(blocks)
        packed_roles(blocks)
        if not isinstance(memo, PolicyMemoController):
            raise ValueError('Policy-aware memo controller required')
        memo.assert_boundary()
        for block in blocks:
            compatible(block._packed_nf4['gate_proj'], block._packed_nf4['up_proj'])
        if getattr(routing, 'PAD', None) != 32 or not hasattr(routing, 'Experts'):
            raise ValueError('Original static row32 Experts routing required')
        if type(diagnostics) is not bool:
            raise ValueError('Explicit diagnostic mode required')
        if dynamic is static or routing.backend is not static or static.BM != 32 or memo.live_bytes:
            raise ValueError('Require distinct backends, static row32 and no live graph')
        memo.configure_from_config(self.current)
        self.memo, self.blocks = memo, blocks
        self.modules = (dynamic, static)
        self.originals = [m.grouped_mm for m in self.modules]
        self.geometry = [{k: getattr(m, k) for k in ('BM','BN','BK','WARPS','WAVES','GROUP_M') if hasattr(m,k)}
                         for m in self.modules]
        self.routing, self.routing_original = routing, routing.forward
        self.alias, self.alias_original = alias, alias.grouped_mm
        self.closed = False
        choose = lambda: self.current
        for module in self.modules:
            directional(module, choose)
            pipeline(module, choose, static_capacity=(module is static))
        row_install(static, routing, choose)
        install_backend(static, choose, self.blocks)
        self.undo_write_once = write_once(static, choose, self.blocks, memo=memo)
        self.row_reuse = RowReuse(static,choose,self.blocks,memo)
        self.pair = PairDispatch(routing, choose, self.blocks, memo=memo, row_reuse=self.row_reuse)
        self.probe = PairAwareProbe(dict(dynamic=dynamic,static=static), self.blocks, memo=memo) if diagnostics else None
        if self.probe:
            self.probe.pair, self.probe.choose = self.pair, choose
        self._memo_install()
        self.routing_installed = routing.forward

    def _memo_install(self):
        self.pre_memo = install_backends(self.memo, self.blocks, self.modules)
        self.installed = [m.grouped_mm for m in self.modules]
        self.alias.grouped_mm = self.modules[0].grouped_mm

    def _boundary(self):
        self.memo.assert_boundary()
        self.row_reuse.boundary()
        self.pair._boundary()
        if self.routing.Experts is not self.pair.installed_experts:
            raise RuntimeError('Remove external Experts wrappers before changing the stack')
        if self.closed or self.memo.live_bytes or (self.probe and self.probe.active):
            raise RuntimeError('Require an open, inactive stack with no live memo graph')
        if (any(m.grouped_mm is not f for m,f in zip(self.modules,self.installed)) or
            self.alias.grouped_mm is not self.installed[0] or self.routing.forward is not self.routing_installed):
            raise RuntimeError('Remove external wrappers before changing the stack')

    def configure(self, config):
        c = normalize(config)
        self._boundary()
        self.row_reuse.configure(c)
        self.memo.configure_from_config(c)
        self.current = c
        self.pair.configure_sources(c)

    def forward(self, original):
        selected = dispatch(original, self.routing)
        def run(block, hidden):
            if self.closed:
                raise RuntimeError('Closed backend stack')
            result=selected(block,hidden) if self.current.get('static') else original(hidden)
            return self.row_reuse.guard(result) if self.current.get('nf4_row_reuse_direction') else result
        return run

    def remove_diagnostics(self):
        self._boundary()
        if self.probe is None:
            return
        for module, original in self.pre_memo:
            module.grouped_mm = original
        self.probe.close()
        self.probe = None
        self._memo_install()

    def assert_throughput_ready(self):
        self._boundary()
        if self.probe is not None:
            raise RuntimeError('Physically remove diagnostic wrappers before throughput timing')

    def close(self):
        self._boundary()
        for module, original in self.pre_memo:
            module.grouped_mm = original
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        self.pair.close()
        self.row_reuse.close()
        self.undo_write_once()
        for module, original, values in zip(self.modules,self.originals,self.geometry):
            module.grouped_mm = original
            for key in ('BM','BN','BK','WARPS','WAVES','GROUP_M'):
                if key in values: setattr(module,key,values[key])
                elif hasattr(module,key): delattr(module,key)
        self.routing.forward = self.routing_original
        self.alias.grouped_mm = self.alias_original
        self.closed = True

