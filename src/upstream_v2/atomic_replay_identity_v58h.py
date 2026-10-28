"""One fixed-winner atomic replay candidate and explicit v57-final migration."""
import copy
import route_input_identity_v57 as prior
import gate_up_identity_candidate_v28 as shared
from route_input_identity_v57 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,AGGREGATE,PREP,INPUT,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
ATOMIC='checkpoint_atomic_route_replay'
V57_WORKER_SHA='87a5551c554986d5572400cd7fd2388bf16ca40ab6c2ef7067b4869c3069596b'
WINNER={'name': 'inputprep_v57_ret23_b1024', 'singleton': True, 'cache': 'pinned', 'retained': 23, 'waves': 4, 'group': 1, 'static': True, 'native_sam_mode': 'native_rms', 'compiler_mode': 'both_dynamic_gemm', 'memo_policy': 'all', 'memo_roles': ['gate_proj', 'up_proj', 'down_proj'], 'memo_bytes': 3623878656, 'memory_tier': 'expanded92', 'nf4_row_reuse_direction': 'both', 'nf4_memo_consume_once': True, 'backward_kernel': {'bn': 128, 'bk': 32, 'warps': 4, 'stages': 1, 'waves': 8, 'group': 1}, 'forward_kernel': {'bn': 64, 'bk': 32, 'warps': 4, 'stages': 1, 'waves': 8, 'group': 1}, 'nf4_row_reuse_waves': {'gate_proj': {'forward_kernel': 32, 'backward_kernel': 32}, 'up_proj': {'forward_kernel': 32, 'backward_kernel': 32}, 'down_proj': {'forward_kernel': 32, 'backward_kernel': 32}}, 'nf4_same_expert_fused64': {'gate_proj': {'backward_kernel': True}, 'up_proj': {'backward_kernel': True}, 'down_proj': {'backward_kernel': True}}, 'nf4_row_scale_cache': {'precision': 'bf16', 'roles': ['gate_proj', 'up_proj', 'down_proj'], 'direction': 'forward', 'max_bytes': 2147483648}, 'expert_pointwise_recompute': {'mode': 'recompute', 'block': 1024}, 'expert_route_aggregate': {'mode': 'both', 'block': 1024}, 'expert_route_backward_prepare': {'mode': 'fused', 'warps': 4}, 'expert_route_input_prepare': {'mode': 'fused', 'block': 1024}}

def normalize(config):
    c=prior.normalize(config); enabled=c.pop(ATOMIC,False)
    if type(enabled) is not bool:raise ValueError('Atomic replay requires an explicit boolean')
    if enabled:
        if prior.computation_identity(c)!=prior.computation_identity(WINNER):
            raise ValueError('Only the fixed v57 winner atomic replay candidate is eligible')
        c[ATOMIC]=True
    return c

def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
    c=normalize(c);return prior.variant(c)+('|atomic_route_replay_v58h' if c.get(ATOMIC) else '')
def family(c):
    c=normalize(c);return ('atomic_route_replay_v58h/' if c.get(ATOMIC) else '')+prior.family(c)
def descriptor(c):
    c=normalize(c);d=prior.descriptor(c)
    d.update(schema='row_reuse_v58_1',compiler_variant=variant(c),
      atomic_route_replay=bool(c.get(ATOMIC)),atomic_route_replay_implementation='routing_replay_candidate_v58',
      atomic_route_metadata_cap_bytes=16*2**20,atomic_route_source_contract='runtime_binding_registry_v58d',
      atomic_route_lifecycle='worker_lifecycle_v58e',atomic_route_observation='replay_observation_v58h',
      atomic_route_scope='one_bound_nonreentrant_checkpoint',atomic_route_cross_update_reuse=False)
    return d

def stamp_continuation(state,c):
    c=normalize(c);state=prior.stamp_continuation(state,c)
    state['identity']['row_reuse_descriptor']=descriptor(c);return state

def validate_resume_config(state,c,*,allow_audited_legacy=False):
    c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
    if schema=='row_reuse_v58_1':
        shared.validate_resume_config(state,c,allow_audited_legacy=False)
        if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Atomic replay descriptor mismatch')
        return 'VALIDATED_ATOMIC_ROUTE_V58'
    if (schema=='row_reuse_v57_1' and not c.get(ATOMIC)
        and not record.get('effective_config',{}).get(ATOMIC)
        and record.get('seal',{}).get('a100_speed_study_v57.py')==V57_WORKER_SHA
        and prior.computation_identity(c)==prior.computation_identity(WINNER)):
        prior.validate_resume_config(state,c,allow_audited_legacy=False)
        return 'EXPLICIT_V57_FINAL_TO_ATOMIC_ROUTE_V58'
    raise ValueError('Require v58 descriptor or this sealed v57 winner source')
