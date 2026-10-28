"""Isolated shared-input identity and explicit audited v27 migration."""
import copy
import memo_transfer_identity_candidate_v27 as prior

FLAG = 'nf4_gate_up_shared_input'


def normalize(config):
    c = prior.normalize(config)
    flag = c.get(FLAG, False)
    if type(flag) is not bool:
        raise ValueError('Shared-input policy must be an explicit boolean')
    if flag and (c.get('static') is not True or c.get('nf4_row_tile', 32) != 32):
        raise ValueError('Shared-input policy requires static row32')
    if not flag:
        c.pop(FLAG, None)
    return c


def computation_identity(config):
    return prior.computation_identity(normalize(config))


def variant(config):
    c = normalize(config)
    return prior.variant(c) + ('|gate_up_shared_input_v1' if c.get(FLAG) else '')


def family(config):
    c = normalize(config)
    return ('gate_up_shared_input_v1/' if c.get(FLAG) else '') + prior.family(c)


def required_seconds(config, rows):
    return 240 if any(r.get('steps', 0) >= 32 and variant(r['config']) == variant(config)
                      for r in rows) else 900


def descriptor(config):
    c = normalize(config)
    return dict(schema='nf4_shared_input_v28_1', policy='gate_up_shared_input_v1' if c.get(FLAG) else 'separate',
                computation_identity=computation_identity(c), compiler_variant=variant(c),
                memo_replay=prior.descriptor(c),
                scope='single_static_experts_forward', static_shape='B2_L_le_1024_row32',
                checkpoint_replay='individual_projection_calls',
                geometry_mismatch='individual_projection_calls',
                dynamic_fallback='original_dispatch_and_directional_geometry',
                weights='independent_frozen_packed_nf4', outputs='two_separate_tensors',
                across_updates_reuse=False, activation_fused=False)


def stamp_continuation(state, config):
    c = normalize(config)
    result = prior.stamp_continuation(state, c)
    result['identity']['shared_input_descriptor'] = descriptor(c)
    return result


def validate_resume_config(state, config, *, allow_audited_legacy=False):
    c = normalize(config)
    saved = normalize(state['identity']['effective_config'])
    if computation_identity(saved) != computation_identity(c):
        raise ValueError('Resume shared-input policy or full configuration differs')
    prior.validate_resume_config(state, c, allow_audited_legacy=False)
    record = state['identity']
    if 'shared_input_descriptor' in record:
        if record['shared_input_descriptor'] != descriptor(saved):
            raise ValueError('Corrupt shared-input descriptor')
        return 'VALIDATED_V28_DESCRIPTOR'
    if not allow_audited_legacy or saved.get(FLAG):
        raise ValueError('Explicit audited v27 separate-policy migration required')
    return 'EXPLICIT_AUDITED_V27_SEPARATE_MIGRATION'


def paired_hypotheses(root):
    """Hypotheses only; no GPU parent admission or scheduling authorization."""
    result = []
    for base in prior.four_grid(normalize(root)):
        for enabled in (False, True):
            c = copy.deepcopy(base)
            c[FLAG] = enabled
            c['name'] += '_shared_%d' % enabled
            result.append(normalize(c))
    return result
