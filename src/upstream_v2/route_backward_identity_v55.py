"""Explicit backward preparation identity; audited v54 source migration only."""
import copy
import route_aggregate_identity_v53 as prior
import gate_up_identity_candidate_v28 as shared
from route_aggregate_identity_v53 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,AGGREGATE,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
PREP='expert_route_backward_prepare'
V54_WORKER_SHA='77f011c057e2b14f1f9b216f5b35d7783e8b574179698b2ce0ae55f7932f16b0'
def normalize(config):
 c=prior.normalize(config);raw=c.pop(PREP,None)
 if raw is None:return c
 if not isinstance(raw,dict) or set(raw)!={'mode','warps'} or raw['mode']!='fused' or type(raw['warps']) is not int or raw['warps'] not in (4,8):raise ValueError('Explicit fused backward preparation and warps4/8 required')
 if not c.get('static') or c.get('nf4_row_tile',32)!=32 or c.get('nf4_gate_up_shared_input'):raise ValueError('Backward preparation requires separate static row32 routing')
 c[PREP]=dict(raw);return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);p=c.get(PREP)
 return prior.variant(c)+('|route_backward_prepare_v55:w'+str(p['warps']) if p else '')
def family(c):
 c=normalize(c);c.pop(PREP,None);return prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v55_1',compiler_variant=variant(c),route_backward_prepare_policy=copy.deepcopy(c.get(PREP,{})),route_backward_prepare_implementation='route_backward_prepare_v55',route_backward_bf16_product_rounding=True,route_backward_reduction_tree_may_differ=True,route_backward_cross_update_cache=False)
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
 if schema=='row_reuse_v55_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Backward preparation descriptor mismatch')
  return 'VALIDATED_ROUTE_BACKWARD_V55'
 if schema=='row_reuse_v53_1' and record.get('seal',{}).get('a100_speed_study_v54.py')==V54_WORKER_SHA and PREP not in c and PREP not in record.get('effective_config',{}):
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy);return 'EXPLICIT_V54_TO_ROUTE_BACKWARD_V55'
 raise ValueError('Missing backward preparation descriptor or unaudited source')
