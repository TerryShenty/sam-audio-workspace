"""Single forward constant-codebook candidate; explicit sealed v59 migration."""
import copy
import atomic_replay_identity_v58h as prior
import gate_up_identity_candidate_v28 as shared
from atomic_replay_identity_v58h import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,AGGREGATE,PREP,INPUT,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
CODEBOOK='nf4_forward_const_codebook'
WINNER=copy.deepcopy(prior.WINNER)
V59_WORKER_SHA='7535ecc451a3c89c19c21eac997c2ad574bf0a3fd9a58227aaf077fa700d01fb'
def normalize(config):
 raw=dict(config);enabled=raw.pop(CODEBOOK,False)
 if type(enabled) is not bool:raise ValueError('Explicit codebook policy boolean required')
 if raw.get(prior.ATOMIC,False):raise ValueError('Atomic replay direction is closed')
 c=prior.normalize(raw)
 if enabled:
  if prior.computation_identity(c)!=prior.computation_identity(WINNER):
   raise ValueError('Only the fixed v57 winner forward codebook candidate is prepared')
  c[CODEBOOK]=True
 return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);enabled=c.pop(CODEBOOK,False)
 return prior.variant(c)+('|forward_const_codebook_v60' if enabled else '')
def family(c):
 c=normalize(c);enabled=c.pop(CODEBOOK,False)
 return ('forward_const_codebook_v60/' if enabled else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='row_reuse_v60_1',compiler_variant=variant(c),
  const_codebook=bool(c.get(CODEBOOK)),const_codebook_implementation='nf4_forward_const_codebook_v60',
  const_codebook_scope='forward_only_bound_actual_16_fp32_entries',
  const_codebook_source_contract='live_runtime_binding_v60')
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c)
 state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if allow_audited_legacy:raise ValueError('Only explicit sealed v59 source permitted')
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v60_1':
  shared.validate_resume_config(state,c,allow_audited_legacy=False)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Codebook descriptor mismatch')
  return 'VALIDATED_CONST_CODEBOOK_V60'
 if (not c.get(CODEBOOK) and not record.get('effective_config',{}).get(CODEBOOK)
     and record.get('seal',{}).get('a100_speed_study_v59.py')==V59_WORKER_SHA
     and prior.computation_identity(c)==prior.computation_identity(WINNER)):
  prior.validate_resume_config(state,c,allow_audited_legacy=False)
  return 'EXPLICIT_V59_FINAL_TO_CONST_CODEBOOK_V60'
 raise ValueError('Require v60 descriptor or exact sealed v59 winner source')
