"""Isolated compiler launch tuning for existing two-tile BK32 arithmetic."""
import copy,hashlib,json
import row_waves_identity_v36 as prior
import gate_up_identity_candidate_v28 as shared_identity
from projection_launch_candidate_v25 import select,ROLES,SIDES
FLAG=prior.FLAG
WAVES=prior.WAVES
TUNING='nf4_row_reuse_tuning'
has_bk64=prior.has_bk64
has_extended_waves=prior.has_extended_waves

def normalize(config):
 c=prior.normalize(config);raw=c.pop(TUNING,{})
 if not isinstance(raw,dict) or set(raw)-set(ROLES):raise ValueError('Unknown row tuning role')
 cleaned={}
 for role,sides in raw.items():
  if not isinstance(sides,dict) or set(sides)-set(SIDES):raise ValueError('Unknown row tuning side')
  for side,overrides in sides.items():
   if not isinstance(overrides,dict) or set(overrides)-{'warps','stages'}:raise ValueError('Only compiler warp/stage overrides')
   if any(type(v) is not int for v in overrides.values()):raise ValueError('Integer compiler options required')
   forward=side=='forward_kernel';base=prior.geometry(c,role,forward)
   values=dict(warps=base['warps'],stages=base['stages']);values.update(overrides)
   if (values['warps'],values['stages']) not in ((4,1),(4,2),(8,1)):raise ValueError('Only separately prepared stages2 or warps8')
   change={k:v for k,v in values.items() if v!=base[k]}
   if not change:continue
   if c.get(FLAG) not in ('both','forward' if forward else 'backward') or c.get('nf4_row_reuse_tiles',2)!=2 or c.get('nf4_gate_up_shared_input') or base['bk']!=32:
    raise ValueError('Compiler tuning requires active separate two-tile BK32 direction')
   cleaned.setdefault(role,{})[side]=change
 if cleaned:c[TUNING]=cleaned
 return c

def has_tuning(c):return bool(normalize(c).get(TUNING))
def geometry(c,role,forward):
 g=prior.geometry(c,role,forward);g.update(c.get(TUNING,{}).get(role,{}).get('forward_kernel' if forward else 'backward_kernel',{}));return g

def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)
 if c.get(TUNING):base+='|row_tuning_v38:'+hashlib.sha256(json.dumps(c[TUNING],sort_keys=True).encode()).hexdigest()
 return base

def family(c):
 c=normalize(c);return ('row_tuning_v38/' if c.get(TUNING) else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v38_1',compiler_variant=variant(c),static_tuning_overrides=copy.deepcopy(c.get(TUNING,{})),static_launches={r:{s:geometry(c,r,s=='forward_kernel') for s in SIDES} for r in ROLES});return d

def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state

def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v38_1':
  shared_identity.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Row compiler tuning descriptor mismatch')
  return 'VALIDATED_ROW_TUNING_V38'
 if not c.get(TUNING) and record.get('seal',{}).get('a100_speed_study_v37.py')=='f54d32b8feff925af091beb53e098ce602425a14f1ecaf62b394fffd5933e34b':
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V37_TO_TUNING_V38'
 raise ValueError('Missing row tuning descriptor or unaudited source')
