"""Future isolated backend stack. Never imported by the running v25 worker.

Diagnostics are inserted before memo, then physically removed before timing.
CPU dispatch tests cannot establish GPU numerical safety or speed benefit.
"""
from directional_backend_v9 import install as directional
from pipeline_launch_candidate_v12 import install as pipeline
from row_tile_runtime_v23 import install as row_install, dispatch
from projection_backend_candidate_v25 import install_backend, packed_roles
from static_write_once_candidate_v1 import install as write_once
from projection_diagnostic_candidate_v1 import ProjectionProbe
from nf4_checkpoint_memo_v10 import install_backends
from write_once_identity_candidate_v26 import normalize, stamp_continuation, validate_resume_config


class BackendStack:
    def __init__(self, dynamic, static, routing, alias, blocks, memo, config, *, diagnostics=False):
        self.current = normalize(config)
        packed_roles(blocks)
        if type(diagnostics) is not bool:
            raise ValueError('Explicit diagnostic mode required')
        if dynamic is static or routing.backend is not static or static.BM != 32 or memo.live_bytes:
            raise ValueError('Require distinct backends, static row32 and no live graph')
        self.memo, self.blocks = memo, list(blocks)
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
        self.probe = ProjectionProbe(dict(dynamic=dynamic,static=static), self.blocks, memo=memo) if diagnostics else None
        self._memo_install()
        self.routing_installed = routing.forward

    def _memo_install(self):
        self.pre_memo = install_backends(self.memo, self.blocks, self.modules)
        self.installed = [m.grouped_mm for m in self.modules]
        self.alias.grouped_mm = self.modules[0].grouped_mm

    def _boundary(self):
        if self.closed or self.memo.live_bytes or (self.probe and self.probe.active):
            raise RuntimeError('Require an open, inactive stack with no live memo graph')
        if (any(m.grouped_mm is not f for m,f in zip(self.modules,self.installed)) or
            self.alias.grouped_mm is not self.installed[0] or self.routing.forward is not self.routing_installed):
            raise RuntimeError('Remove external wrappers before changing the stack')

    def configure(self, config):
        c = normalize(config)
        self._boundary()
        self.memo.configure(c.get('memo_roles',()), c.get('memo_bytes',0), c.get('memo_verify',False))
        self.current = c

    def forward(self, original):
        selected = dispatch(original, self.routing)
        def run(block, hidden):
            if self.closed:
                raise RuntimeError('Closed backend stack')
            return selected(block,hidden) if self.current.get('static') else original(hidden)
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
        self.undo_write_once()
        for module, original, values in zip(self.modules,self.originals,self.geometry):
            module.grouped_mm = original
            for key in ('BM','BN','BK','WARPS','WAVES','GROUP_M'):
                if key in values: setattr(module,key,values[key])
                elif hasattr(module,key): delattr(module,key)
        self.routing.forward = self.routing_original
        self.alias.grouped_mm = self.alias_original
        self.closed = True


def effective_resume(state, manifest, current_identity, current_job, restore, snapshot,
                     step, pairs, save, tick, *, config, allow_audited_legacy=False,
                     updates=32, device_kind='cuda'):
    from historical_resume_v22 import effective_resume as prior_resume
    migration = validate_resume_config(state,config,allow_audited_legacy=allow_audited_legacy)
    def save_checked(post):
        saved = stamp_continuation(post,post['identity']['effective_config'])
        validate_resume_config(saved,config)
        save(saved)
    report = prior_resume(state,manifest,current_identity,current_job,restore,snapshot,
                          step,pairs,save_checked,tick,updates=updates,device_kind=device_kind)
    report.update(output_initialization_identity_validation=migration,
                  saved_output_initialization_descriptor=True)
    return report
