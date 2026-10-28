"""Explicit grouped row-pair traversal identity; only audited v40 migration."""
import copy,hashlib,json
import row_fused64_identity_v39 as prior
import gate_up_identity_candidate_v28 as shared_identity
from projection_launch_candidate_v25 import select,ROLES,SIDES
FLAG=prior.FLAG
WAVES=prior.WAVES
TUNING=prior.TUNING
FUSION=prior.FUSION
GROUPED='nf4_row_pair_groups'
has_bk64=prior.has_bk64
has_extended_waves=prior.has_extended_waves
has_tuning=prior.has_tuning
has_fusion=prior.has_fusion
geometry=prior.geometry
def normalize(config):
 c=prior.normalize(config);raw=c.pop(GROUPED,{})
 if not isinstance(raw,dict) or set(raw)-set(ROLES):raise ValueError('Unknown grouped role')
 cleaned={}
 for role,sides in raw.items():
  if not isinstance(sides,dict) or set(sides)-set(SIDES):raise ValueError('Unknown grouped side')
  for side,group in sides.items():
   if type(group) is not int or group not in (2,4):raise ValueError('Require group of two or four row pairs')
   g=geometry(c,role,side=='forward_kernel')
   if c.get(FLAG) not in ('both',side.split('_')[0]) or c.get('nf4_row_reuse_tiles',2)!=2 or c.get('nf4_gate_up_shared_input') or g['bn'] not in (64,128) or (g['bk'],g['warps'],g['stages'])!=(32,4,1) or c.get(FUSION,{}).get(role,{}).get(side):raise ValueError('Grouped traversal requires separate nonfused BK32 row-pair warps4 stages1')
   cleaned.setdefault(role,{})[side]=group
 if cleaned:c[GROUPED]=cleaned
 return c
def has_grouped(c):return bool(normalize(c).get(GROUPED))
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)
 if c.get(GROUPED):base+='|row_pair_grouped_v41:'+hashlib.sha256(json.dumps(c[GROUPED],sort_keys=True).encode()).hexdigest()
 return base
def family(c):
 c=normalize(c);return ('grouped_v41/' if c.get(GROUPED) else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v41_1',compiler_variant=variant(c),row_pair_groups=copy.deepcopy(c.get(GROUPED,{})),row_pair_traversal='grouped_row_pairs_column_major_v41');return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v41_1':
  shared_identity.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Grouped descriptor mismatch')
  return 'VALIDATED_ROW_GROUPED_V41'
 if not c.get(GROUPED) and record.get('seal',{}).get('a100_speed_study_v40.py')=='066d6f4ee725ce7bc0d085f5274da2fb71306f992d9d2540f2e0a95e35c37dc1':
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V40_TO_GROUPED_V41'
 raise ValueError('Missing grouped descriptor or unaudited source')
