"""Isolated successor identity; never imported by the sealed v26 worker."""
import copy
import write_once_identity_candidate_v26 as prior

FLAG = 'nf4_memo_consume_once'


def normalize(config):
    c = prior.normalize(config)
    flag = c.get(FLAG, False)
    if type(flag) is not bool:
        raise ValueError('Memo transfer policy must be an explicit boolean')
    if not flag:
        c.pop(FLAG, None)
    return c


def computation_identity(config):
    return prior.computation_identity(normalize(config))


def variant(config):
    c = normalize(config)
    return prior.variant(c) + ('|memo_consume_once_v1' if c.get(FLAG) else '')


def family(config):
    c = normalize(config)
    return ('memo_consume_once_v1/' if c.get(FLAG) else '') + prior.family(c)


def required_seconds(config, rows):
    return 240 if any(r.get('steps', 0) >= 32 and variant(r['config']) == variant(config)
                      for r in rows) else 900


def descriptor(config):
    c = normalize(config)
    return dict(schema='nf4_memo_replay_v27_1',
                policy='consume_once_v1' if c.get(FLAG) else 'clone',
                computation_identity=computation_identity(c), compiler_variant=variant(c),
                output_initialization=prior.descriptor(c),
                first_store='detached_clone', scope_reservation='until_scope_release',
                repeated_replay='recompute' if c.get(FLAG) else 'clone',
                across_updates_reuse=False)


def stamp_continuation(state, config):
    c = normalize(config)
    result = prior.stamp_continuation(state, c)
    result['identity']['memo_replay_descriptor'] = descriptor(c)
    return result


def validate_resume_config(state, config, *, allow_audited_legacy=False):
    c = normalize(config)
    saved = normalize(state['identity']['effective_config'])
    if computation_identity(saved) != computation_identity(c):
        raise ValueError('Resume memo policy or complete configuration differs')
    # Even explicit legacy migration must carry the complete v26 descriptors.
    # Missing/corrupt descriptors cannot be replaced by a new policy declaration.
    prior.validate_resume_config(state, c, allow_audited_legacy=False)
    identity = state['identity']
    if 'memo_replay_descriptor' in identity:
        if identity['memo_replay_descriptor'] != descriptor(saved):
            raise ValueError('Corrupt memo replay descriptor')
        return 'VALIDATED_V27_DESCRIPTOR'
    if not allow_audited_legacy or saved.get(FLAG):
        raise ValueError('Explicit audited v26 clone-policy migration required')
    return 'EXPLICIT_AUDITED_V26_CLONE_MIGRATION'


def four_grid(root):
    """Four hypotheses, NOT a fresh-parent admission or GPU scheduling decision."""
    c = normalize(root)
    if not c.get('static') or c.get('nf4_row_tile', 32) != 32:
        raise ValueError('Four-grid comparison requires an explicit static row32 root')
    result = []
    for once, consume in ((False, False), (False, True), (True, False), (True, True)):
        candidate = copy.deepcopy(c)
        candidate.update(nf4_static_write_once=once, nf4_memo_consume_once=consume)
        candidate['name'] = 'memo27_once_%d_consume_%d' % (once, consume)
        result.append(normalize(candidate))
    return result
