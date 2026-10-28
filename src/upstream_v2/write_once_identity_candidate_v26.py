"""Identity contract for an isolated future worker; does not launch work."""
import copy
from projection_identity_candidate_v25 import (
    normalize as prior_normalize, computation_identity as prior_identity,
    variant as prior_variant, checkpoint_descriptor as projection_descriptor,
    validate_checkpoint_descriptor as validate_projection)
from successor_design_v25 import family as prior_family


def normalize(config):
    c = prior_normalize(config)
    flag = c.get('nf4_static_write_once', False)
    if type(flag) is not bool:
        raise ValueError('Write-once policy must be an explicit boolean')
    if flag and (not c.get('static') or c.get('nf4_row_tile', 32) != 32):
        raise ValueError('Write-once requires static row32')
    if not flag:
        c.pop('nf4_static_write_once', None)
    return c


def computation_identity(config):
    return prior_identity(normalize(config))


def variant(config):
    c = normalize(config)
    return prior_variant(c) + ('|write_once_v1' if c.get('nf4_static_write_once') else '')


def family(config):
    c = normalize(config)
    return 'write_once_row32' if c.get('nf4_static_write_once') else prior_family(c)


def required_seconds(config, rows):
    return 240 if any(r.get('steps', 0) >= 32 and variant(r['config']) == variant(config)
                      for r in rows) else 900


def descriptor(config):
    c = normalize(config)
    return dict(schema='nf4_output_initialization_v26_1',
                policy='write_once_v1' if c.get('nf4_static_write_once') else 'original',
                computation_identity=computation_identity(c), compiler_variant=variant(c),
                projection=projection_descriptor(c), dynamic_fallback='original',
                caller_padding_zero_preserved=True)


def stamp_continuation(state, config):
    result = dict(state)
    result['identity'] = copy.deepcopy(state['identity'])
    result['identity']['effective_config'] = copy.deepcopy(config)
    result['identity']['projection_descriptor'] = projection_descriptor(config)
    result['identity']['output_initialization_descriptor'] = descriptor(config)
    return result


def validate_resume_config(state, config, *, allow_audited_legacy=False):
    saved = state['identity']['effective_config']
    if computation_identity(saved) != computation_identity(config):
        raise ValueError('Resume computation differs from requested configuration')
    identity = state['identity']
    if 'output_initialization_descriptor' in identity:
        if identity['output_initialization_descriptor'] != descriptor(saved):
            raise ValueError('Corrupt output-initialization descriptor')
        validate_projection(saved, identity.get('projection_descriptor'))
        return 'VALIDATED_V26_DESCRIPTOR'
    if not allow_audited_legacy or normalize(saved).get('nf4_static_write_once'):
        raise ValueError('Missing write-once descriptor; legacy migration must be explicit')
    if 'projection_descriptor' in identity:
        validate_projection(saved, identity['projection_descriptor'])
    elif normalize(saved).get('projection_kernels'):
        raise ValueError('Projection overrides require their descriptor')
    return 'EXPLICIT_AUDITED_LEGACY_MIGRATION'
