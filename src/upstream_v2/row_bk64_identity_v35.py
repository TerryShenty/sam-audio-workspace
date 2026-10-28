"""Row32 groups of two or four; explicit same-policy v32 migration."""
import row_bk64_pair_identity_v35 as prior
import row_quad_identity_v33 as old_identity
from projection_launch_candidate_v25 import select,ROLES
import hashlib,json
FLAG=prior.FLAG
TILES='nf4_row_reuse_tiles'
def has_bk64(c):
 direction=c.get(FLAG,'off')
 return any(direction in ('both','forward' if forward else 'backward') and select(c,role,forward)['bk']==64 for role in ROLES for forward in (True,False))
def normalize(config):
 c=prior.normalize(config);n=c.get(TILES,2)
 if type(n) is not int or n not in (2,4):raise ValueError('Prepared row groups are two or four tiles')
 if n==4 and not c.get(FLAG):raise ValueError('Quad requires an active row-reuse direction')
 if has_bk64(c) and (n!=2 or c.get('nf4_gate_up_shared_input')):raise ValueError('BK64 prepared only for separate two-tile reuse')
 if n==2:c.pop(TILES,None)
 return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)+('|row_quad_v1' if c.get(TILES)==4 else '')
 if has_bk64(c):
  geometries=[(role,forward,select(c,role,forward)) for role in ROLES for forward in (True,False)]
  base+='|row_bk64_v35:'+hashlib.sha256(json.dumps(geometries,sort_keys=True).encode()).hexdigest()
 return base
def family(c):
 c=normalize(c);return ('row_quad_v1/' if c.get(TILES)==4 else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v35_1',row_pair=32*c.get(TILES,2),row_tiles=c.get(TILES,2),compiler_variant=variant(c),scope='single_kernel_adjacent_row32_tiles_same_expert')
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v35_1':
  prior.prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Row group resume descriptor mismatch')
  return 'VALIDATED_ROW_BK64_V35'
 if not has_bk64(c) and record.get('seal',{}).get('a100_speed_study_v34.py')=='9c0216184e527efb4ac104c4ccf32b29ddbbbd0d801f55bb6883033df4a4e689':
  old_identity.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V34_BK32_MIGRATION'
 raise ValueError('Missing BK64 descriptor or unaudited source')
