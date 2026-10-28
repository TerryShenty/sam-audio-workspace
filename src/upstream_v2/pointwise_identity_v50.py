"""Explicit graph-local pointwise recomputation identity and audited v49 migration."""
import copy
import finite_guard_identity_v48 as prior
import gate_up_identity_candidate_v28 as shared
from finite_guard_identity_v48 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
POINTWISE='expert_pointwise_recompute'
V49_WORKER_SHA='49474de7de79b33c7046d6c6d388d84c9b08307576049771f0fb61746854a26b'
def normalize(config):
 c=prior.normalize(config);raw=c.pop(POINTWISE,None)
 if raw is None:return c
 if not isinstance(raw,dict) or set(raw)!={'mode','block'} or raw['mode']!='recompute' or type(raw['block']) is not int or raw['block'] not in (256,512,1024):raise ValueError('Explicit pointwise recompute mode and block required')
 if not c.get('static') or c.get('nf4_row_tile',32)!=32 or c.get('nf4_gate_up_shared_input'):raise ValueError('Pointwise candidate requires separate static row32 routing')
 c[POINTWISE]=dict(mode='recompute',block=raw['block']);return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);return prior.variant(c)+('|pointwise_recompute_v50:'+str(c[POINTWISE]['block']) if POINTWISE in c else '')
def family(c):
 c=normalize(c);c.pop(POINTWISE,None);return prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v50_1',compiler_variant=variant(c),pointwise_policy=copy.deepcopy(c.get(POINTWISE,{})),pointwise_implementation='pointwise_recompute_v50',pointwise_saved_activation=False if POINTWISE in c else True,pointwise_bf16_intermediates=True,pointwise_cross_update_cache=False)
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
 if schema=='row_reuse_v50_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Pointwise descriptor mismatch')
  return 'VALIDATED_POINTWISE_V50'
 if schema=='row_reuse_v48_1' and record.get('seal',{}).get('a100_speed_study_v49.py')==V49_WORKER_SHA and POINTWISE not in c and POINTWISE not in record.get('effective_config',{}):
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy);return 'EXPLICIT_V49_TO_POINTWISE_V50'
 raise ValueError('Missing pointwise descriptor or unaudited source')
