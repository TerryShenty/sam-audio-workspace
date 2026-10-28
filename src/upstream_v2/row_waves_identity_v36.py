"""Isolated static row-pair CTA scheduling; underlying arithmetic is unchanged."""
import copy,hashlib,json
import row_bk64_identity_v35 as prior
import gate_up_identity_candidate_v28 as shared_identity
from projection_launch_candidate_v25 import select,ROLES,SIDES
FLAG=prior.FLAG
WAVES='nf4_row_reuse_waves'
has_bk64=prior.has_bk64

def normalize(config):
 c=prior.normalize(config);raw=c.pop(WAVES,{})
 if not isinstance(raw,dict) or set(raw)-set(ROLES):raise ValueError('Row wave overrides require known projection roles')
 cleaned={}
 for role,sides in raw.items():
  if not isinstance(sides,dict) or set(sides)-set(SIDES):raise ValueError('Unknown row wave direction')
  for side,waves in sides.items():
   if type(waves) is not int or waves not in (2,4,8,16,32):raise ValueError('Prepared CTA wave counts are 2,4,8,16,32')
   forward=side=='forward_kernel';base=select(c,role,forward)
   if waves==base['waves']:continue
   if c.get(FLAG) not in ('both','forward' if forward else 'backward') or c.get('nf4_row_reuse_tiles',2)!=2 or c.get('nf4_gate_up_shared_input') or base['bk']!=32:
    raise ValueError('Wave overrides require active separate two-tile BK32 direction')
   cleaned.setdefault(role,{})[side]=waves
 if cleaned:c[WAVES]=cleaned
 return c

def has_extended_waves(c):return bool(normalize(c).get(WAVES))
def geometry(c,role,forward):
 g=select(c,role,forward);side='forward_kernel' if forward else 'backward_kernel'
 if side in c.get(WAVES,{}).get(role,{}):g['waves']=c[WAVES][role][side]
 return g

def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);base=prior.variant(c)
 if c.get(WAVES):base+='|row_waves_v36:'+hashlib.sha256(json.dumps(c[WAVES],sort_keys=True).encode()).hexdigest()
 return base

def family(c):
 c=normalize(c);return ('row_waves_v36/' if c.get(WAVES) else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v36_1',compiler_variant=variant(c),static_wave_overrides=copy.deepcopy(c.get(WAVES,{})),static_launches={r:{s:geometry(c,r,s=='forward_kernel') for s in SIDES} for r in ROLES})
 return d

def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state

def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v36_1':
  shared_identity.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Row waves descriptor mismatch')
  return 'VALIDATED_ROW_WAVES_V36'
 if not c.get(WAVES) and record.get('seal',{}).get('a100_speed_study_v35.py')=='72c5eefc81fbca9720ae43177fad9de723a247561eaf07a0c482ebc76c5cd294':
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V35_TO_WAVES_V36'
 raise ValueError('Missing row waves descriptor or unaudited source')
