"""Explicit finite-guard execution identity and audited v47 continuation."""
import copy
import row_scale_fused_identity_v46 as prior
import gate_up_identity_candidate_v28 as shared
from row_scale_fused_identity_v46 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
GUARD='finite_guard_mode'
V47_WORKER_SHA='32ebca7ee20d1acc69c441a1cad0ae789015f8126eb133531df07ae0fdc75056'
def normalize(config):
 c=prior.normalize(config);mode=c.pop(GUARD,None)
 if mode is None or mode=='reference':return c
 if type(mode) is not str or mode not in ('combined','grouped'):raise ValueError('Unknown finite guard mode')
 c[GUARD]=mode;return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)
 return base+('|finite_guard_v48:'+c[GUARD] if GUARD in c else '')
def family(c):
 # In particular, an added guard cannot reset the FP32 memory failure family.
 c=normalize(c);c.pop(GUARD,None);return prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v48_1',compiler_variant=variant(c),finite_guard_mode=c.get(GUARD,'reference'),finite_guard_implementation='finite_guard_component_v48',finite_guard_before_optimizer=True,finite_guard_tensor_cache=False)
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity'];schema=record.get('row_reuse_descriptor',{}).get('schema')
 if schema=='row_reuse_v48_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Finite guard descriptor mismatch')
  return 'VALIDATED_FINITE_GUARD_V48'
 if schema=='row_reuse_v46_1' and record.get('seal',{}).get('a100_speed_study_v47.py')==V47_WORKER_SHA and GUARD not in c and GUARD not in record.get('effective_config',{}):
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V47_TO_FINITE_GUARD_V48'
 raise ValueError('Missing finite guard descriptor or unaudited source')
