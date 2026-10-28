"""CPU-only per-projection launch selection. Not installed in any GPU worker."""
import copy
from pipeline_launch_candidate_v12 import launch_config
ROLES=('gate_proj','up_proj','down_proj')
SIDES=('forward_kernel','backward_kernel')
FIELDS={'bn','bk','warps','waves','group','stages'}

def validate(config):
    overrides=config.get('projection_kernels',{})
    if not isinstance(overrides,dict):raise ValueError('Projection overrides must be a mapping')
    if overrides and (not config.get('static',False) or config.get('nf4_row_tile',32)!=32):
        raise ValueError('Initial per-projection candidate requires original static row32')
    if set(overrides)-set(ROLES):raise ValueError('Unknown projection role')
    for role,sides in overrides.items():
        if not isinstance(sides,dict) or set(sides)-set(SIDES):raise ValueError('Unknown direction')
        for side,values in sides.items():
            if not isinstance(values,dict) or set(values)-FIELDS:raise ValueError('Unknown launch setting')
            if any(type(v) is not int for v in values.values()):raise ValueError('Launch values must be integers')
            merged=dict(config);merged[side]=dict(config.get(side,{}),**values)
            launch_config(merged,side=='forward_kernel')
    return overrides

def select(config,role,transposed):
    if role not in ROLES or type(transposed) is not bool:raise ValueError('Explicit role and direction required')
    overrides=validate(config);side='forward_kernel' if transposed else 'backward_kernel'
    merged=dict(config);merged[side]=dict(config.get(side,{}),**overrides.get(role,{}).get(side,{}))
    return launch_config(merged,transposed)

def effective_identity(config):
    """Launch suffix only: future worker must combine with full computation identity."""
    return tuple((role,transposed,tuple(sorted(select(config,role,transposed).items())))
                 for role in ROLES for transposed in (True,False))

def proposals(incumbent):
    validate(incumbent)
    if incumbent.get('nf4_row_tile',32)!=32 or not incumbent.get('static',False):
        raise ValueError('Per-projection candidates need an admitted row32 parent')
    changes=[('down_proj','forward_kernel',dict(bn=128)),
             ('gate_proj','forward_kernel',dict(bn=128)),
             ('up_proj','forward_kernel',dict(bn=128)),
             ('down_proj','backward_kernel',dict(bn=64)),
             ('gate_proj','backward_kernel',dict(waves=4)),
             ('up_proj','backward_kernel',dict(waves=4)),
             ('down_proj','forward_kernel',dict(bk=64)),
             ('down_proj','backward_kernel',dict(warps=8))]
    seen={effective_identity(incumbent)}
    for role,side,values in changes:
        c=copy.deepcopy(incumbent)
        overrides=c.setdefault('projection_kernels',{})
        overrides.setdefault(role,{}).setdefault(side,{}).update(values)
        ident=effective_identity(c)
        if ident in seen:continue
        seen.add(ident);c['name']=f'projection_v25_{role}_{side}_{len(seen)-1}'
        yield c
