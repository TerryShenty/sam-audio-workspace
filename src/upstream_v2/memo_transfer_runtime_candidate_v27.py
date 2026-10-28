"""Isolated clone/consume adapter with graph-lifetime mutation protection.

Only CPU adapter acceptance is intended here. No running worker imports this.
"""
import nf4_checkpoint_memo_v10 as base
from memo_transfer_candidate_v1 import TransferMemoController, _Scope, _ReusableContext
from write_once_runtime_candidate_v26 import BackendStack as PriorStack
from memo_transfer_identity_candidate_v27 import (FLAG, normalize, stamp_continuation,
                                                 validate_resume_config)


class _TrackedScope(_Scope):
    def __init__(self, owner):
        super().__init__(owner)
        owner.outstanding_scopes += 1

    def __del__(self):
        super().__del__()
        self.owner.outstanding_scopes -= 1


class PolicyMemoController(TransferMemoController):
    def __init__(self, roles=('gate_proj', 'up_proj'), max_bytes=2**30,
                 verify_recompute=False):
        self.consume_once = False
        self.outstanding_scopes = 0
        super().__init__(roles, max_bytes, verify_recompute)

    def assert_boundary(self):
        if (self.outstanding_scopes or getattr(self, 'active_contexts', 0) or
                getattr(self, 'live_bytes', 0)):
            raise RuntimeError('Release every memo graph/context before policy or wrapper changes')

    def configure(self, roles, max_bytes, verify_recompute=False):
        self.assert_boundary()
        if (type(max_bytes) is not int or max_bytes < 0 or
                type(verify_recompute) is not bool or not set(roles) <= base.ROLES):
            raise ValueError('Invalid memo budget, roles or verification policy')
        super().configure(roles, max_bytes, verify_recompute)

    def configure_from_config(self, config):
        c = normalize(config)
        self.configure(c.get('memo_roles', ()), c.get('memo_bytes', 0), c.get('memo_verify', False))
        self.consume_once = c.get(FLAG, False)

    def contexts(self):
        scope = _TrackedScope(self)
        return _ReusableContext(scope, False), _ReusableContext(scope, True)

    def wrap(self, original, role_by_packed_id):
        clone = base.MemoController.wrap(self, original, role_by_packed_id)
        consume = TransferMemoController.wrap(self, original, role_by_packed_id)
        def grouped_mm(*args):
            return (consume if self.consume_once else clone)(*args)
        return grouped_mm

    def statistics(self):
        result = super().statistics()
        result.update(replay_policy='consume_once_v1' if self.consume_once else 'clone',
                      outstanding_scopes=self.outstanding_scopes)
        return result


class BackendStack(PriorStack):
    def __init__(self, dynamic, static, routing, alias, blocks, memo, config, *, diagnostics=False):
        c = normalize(config)
        if not isinstance(memo, PolicyMemoController):
            raise ValueError('Explicit policy-aware memo controller required')
        memo.assert_boundary()
        memo.configure_from_config(c)
        super().__init__(dynamic, static, routing, alias, blocks, memo, c, diagnostics=diagnostics)

    def _boundary(self):
        self.memo.assert_boundary()
        super()._boundary()

    def configure(self, config):
        c = normalize(config)
        self._boundary()
        self.memo.configure_from_config(c)
        self.current = c


def effective_resume(state, manifest, current_identity, current_job, restore, snapshot,
                     step, pairs, save, tick, *, config, allow_audited_legacy=False,
                     updates=32, device_kind='cuda'):
    from historical_resume_v22 import effective_resume as prior_resume
    if updates != 32 or type(updates) is not int:
        raise ValueError('Successor acceptance requires 32 effective continuation updates')
    migration = validate_resume_config(state, config, allow_audited_legacy=allow_audited_legacy)
    def save_checked(post):
        saved = stamp_continuation(post, post['identity']['effective_config'])
        validate_resume_config(saved, config)
        save(saved)
    report = prior_resume(state, manifest, current_identity, current_job, restore, snapshot,
                          step, pairs, save_checked, tick, updates=updates, device_kind=device_kind)
    report.update(memo_replay_identity_validation=migration, saved_memo_replay_descriptor=True)
    return report
