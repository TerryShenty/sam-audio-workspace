"""Explicit route aggregation identity with audited v52 checkpoint migration."""
import copy
import pointwise_identity_v50 as prior
import gate_up_identity_candidate_v28 as shared
from pointwise_identity_v50 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
AGGREGATE='expert_route_aggregate'
V52_WORKER_SHA='a6b0ba710c5513b081dc77748c269399991b5d512d35708c4d8beacf15c9d0e0'
def normalize(config):
 c=prior.normalize(config);raw=c.pop(AGGREGATE,None)
 if raw is None:return c
 if not isinstance(raw,dict) or set(raw)!={'mode','block'} or raw['mode'] not in ('weighted_forward','input_backward','both') or type(raw['block']) is not int or raw['block'] not in (256,512,1024):raise ValueError('Explicit aggregate direction and block required')
 if not c.get('static') or c.get('nf4_row_tile',32)!=32 or c.get('nf4_gate_up_shared_input'):raise ValueError('Route aggregation requires separate static row32 routing')
 c[AGGREGATE]=dict(raw);return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);p=c.get(AGGREGATE)
 return prior.variant(c)+('|route_aggregate_v53:'+p['mode']+':'+str(p['block']) if p else '')
def family(c):
 c=normalize(c);c.pop(AGGREGATE,None);return prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v53_1',compiler_variant=variant(c),route_aggregate_policy=copy.deepcopy(c.get(AGGREGATE,{})),route_aggregate_implementation='route_aggregate_v53',route_order='forward_ascending_backward_descending_expert',route_bf16_rounding_boundaries=True,route_cross_update_cache=False)
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
 if schema=='row_reuse_v53_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Route aggregation descriptor mismatch')
  return 'VALIDATED_ROUTE_AGGREGATE_V53'
 if schema=='row_reuse_v50_1' and record.get('seal',{}).get('a100_speed_study_v52.py')==V52_WORKER_SHA and AGGREGATE not in c and AGGREGATE not in record.get('effective_config',{}):
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy);return 'EXPLICIT_V52_TO_ROUTE_AGGREGATE_V53'
 raise ValueError('Missing route aggregation descriptor or unaudited source')
