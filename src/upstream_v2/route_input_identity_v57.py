"""Forward input preparation identity; explicit sealed v56 migration only."""
import copy
import route_backward_identity_v55 as prior
import gate_up_identity_candidate_v28 as shared
from route_backward_identity_v55 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,AGGREGATE,PREP,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
INPUT='expert_route_input_prepare'
V56_WORKER_SHA='72d3c153467d89c5224f0373bd87bd5b541f31d0ee961e136629770a723c95cb'
def normalize(config):
 c=prior.normalize(config);raw=c.pop(INPUT,None)
 if raw is None:return c
 if not isinstance(raw,dict) or set(raw)!={'mode','block'} or raw['mode']!='fused' or type(raw['block']) is not int or raw['block'] not in (256,512,1024):raise ValueError('Explicit fused input preparation and block256/512/1024 required')
 if not c.get('static') or c.get('nf4_row_tile',32)!=32 or c.get('nf4_gate_up_shared_input'):raise ValueError('Input preparation requires separate static row32 routing')
 c[INPUT]=dict(raw);return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);p=c.get(INPUT)
 return prior.variant(c)+('|route_input_prepare_v57:b'+str(p['block']) if p else '')
def family(c):
 c=normalize(c);c.pop(INPUT,None);return prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v57_1',compiler_variant=variant(c),route_input_prepare_policy=copy.deepcopy(c.get(INPUT,{})),route_input_prepare_implementation='route_input_prepare_v57',route_input_padding_fully_written=True,route_input_cross_update_cache=False)
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
 if schema=='row_reuse_v57_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Input preparation descriptor mismatch')
  return 'VALIDATED_ROUTE_INPUT_V57'
 if schema=='row_reuse_v55_1' and record.get('seal',{}).get('a100_speed_study_v56.py')==V56_WORKER_SHA and INPUT not in c and INPUT not in record.get('effective_config',{}):
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy);return 'EXPLICIT_V56_TO_ROUTE_INPUT_V57'
 raise ValueError('Missing input preparation descriptor or unaudited source')
