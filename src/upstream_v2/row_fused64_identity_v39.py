"""Isolated row64 same-expert arithmetic identity with explicit v38 migration."""
import copy,hashlib,json
import row_tuning_identity_v38 as prior
import gate_up_identity_candidate_v28 as shared_identity
from projection_launch_candidate_v25 import select,ROLES,SIDES
FLAG=prior.FLAG
WAVES=prior.WAVES
TUNING=prior.TUNING
FUSION='nf4_same_expert_fused64'
has_bk64=prior.has_bk64
has_extended_waves=prior.has_extended_waves
has_tuning=prior.has_tuning
geometry=prior.geometry
def normalize(config):
 c=prior.normalize(config);raw=c.pop(FUSION,{})
 if not isinstance(raw,dict) or set(raw)-set(ROLES):raise ValueError('Unknown fused64 role')
 cleaned={}
 for role,sides in raw.items():
  if not isinstance(sides,dict) or set(sides)-set(SIDES):raise ValueError('Unknown fused64 side')
  for side,on in sides.items():
   if type(on) is not bool:raise ValueError('Boolean fusion switch required')
   if not on:continue
   g=geometry(c,role,side=='forward_kernel')
   if c.get(FLAG) not in ('both',side.split('_')[0]) or c.get('nf4_row_reuse_tiles',2)!=2 or c.get('nf4_gate_up_shared_input') or g['bn'] not in (64,128) or (g['bk'],g['warps'],g['stages'])!=(32,4,1):raise ValueError('Fused64 requires active separate row-pair BK32 warps4 stages1')
   cleaned.setdefault(role,{})[side]=True
 if cleaned:c[FUSION]=cleaned
 return c
def has_fusion(c):return bool(normalize(c).get(FUSION))
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)
 if c.get(FUSION):base+='|same_expert_fused64_v39:'+hashlib.sha256(json.dumps(c[FUSION],sort_keys=True).encode()).hexdigest()
 return base
def family(c):
 c=normalize(c);return ('fused64_v39/' if c.get(FUSION) else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v39_1',compiler_variant=variant(c),same_expert_fused64=copy.deepcopy(c.get(FUSION,{})),fusion_condition='same_expert_and_full_64_rows',branch_evidence='device_executed_output_tiles');return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v39_1':
  shared_identity.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Fused64 descriptor mismatch')
  return 'VALIDATED_ROW_FUSED64_V39'
 if not c.get(FUSION) and record.get('seal',{}).get('a100_speed_study_v38.py')=='42c874533db817e7954c6b4d663906cbfe7f967591be9ae03ec894afa1c67483':
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V38_TO_FUSED64_V39'
 raise ValueError('Missing fused64 descriptor or unaudited source')
