"""Isolated identity component; no worker or checkpoint loader imports this yet."""
import copy
import hashlib
import json
from successor_design_v23 import computation_identity as prior_identity, normalize as prior_normalize, variant as prior_variant
from projection_launch_candidate_v25 import ROLES, SIDES, validate, select, proposals
from pipeline_launch_candidate_v12 import launch_config


def normalize(config):
    c = prior_normalize(config)
    validate(c)
    overrides = {}
    for role in ROLES:
        for side in SIDES:
            forward = side == 'forward_kernel'
            base = launch_config(c, forward)
            actual = select(c, role, forward)
            difference = {k: v for k, v in actual.items() if base[k] != v}
            if difference:
                overrides.setdefault(role, {})[side] = difference
    if overrides:
        c['projection_kernels'] = overrides
    else:
        c.pop('projection_kernels', None)
    return c


def computation_identity(config):
    # Keep base directional geometry: it still controls dynamic/long fallback.
    # Equal static launches alone do not establish equal complete computation.
    return prior_identity(normalize(config))


def variant(config):
    c = normalize(config)
    base = prior_variant(c)
    if not c.get('projection_kernels'):
        return base
    launches = [(role, forward, select(c, role, forward))
                for role in ROLES for forward in (True, False)]
    suffix = hashlib.sha256(json.dumps(launches, sort_keys=True).encode()).hexdigest()
    return base + '|projection25:' + suffix


def required_seconds(config, rows):
    return 240 if any(r.get('steps', 0) >= 32 and variant(r['config']) == variant(config)
                      for r in rows) else 900


def checkpoint_descriptor(config):
    c = normalize(config)
    return dict(schema='projection_identity_v25_1',
                computation_identity=computation_identity(c),
                compiler_variant=variant(c),
                dynamic_fallback_launches={side: launch_config(c, side == 'forward_kernel')
                                           for side in SIDES},
                static_projection_launches={role: {side: select(c, role, side == 'forward_kernel')
                                                  for side in SIDES} for role in ROLES})


def validate_checkpoint_descriptor(config, descriptor):
    if descriptor != checkpoint_descriptor(config):
        raise ValueError('Checkpoint projection descriptor does not match complete configuration')
    return True


def untested_proposals(incumbent, measured, rejected, planned):
    excluded = {computation_identity(incumbent)}
    excluded.update(computation_identity(r['config']) for r in measured + rejected if r.get('config'))
    excluded.update(computation_identity(c) for c in planned)
    result = []
    for candidate in proposals(incumbent):
        c = normalize(candidate)
        identity = computation_identity(c)
        if identity in excluded:
            continue
        result.append(copy.deepcopy(c))
        excluded.add(identity)
    return result
