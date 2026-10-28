"""Complete row-pair policy identity; no numerical-equivalence requirement."""
import gate_up_identity_candidate_v28 as prior
from projection_launch_candidate_v25 import select,ROLES
FLAG='nf4_row_reuse_direction'
def normalize(config):
 c=prior.normalize(config);direction=c.get(FLAG,'off')
 if direction not in ('off','forward','backward','both'):raise ValueError('Unknown row reuse direction')
 if direction=='off':c.pop(FLAG,None);return c
 if not c.get('static') or c.get('nf4_row_tile',32)!=32:raise ValueError('Static row32 required')
 for role in ROLES:
  for forward in (True,False):
   if direction not in ('both','forward' if forward else 'backward'):continue
   v=select(c,role,forward)
   if v['bn'] not in (64,128) or v['bk']!=32 or v['warps']!=4 or v['stages']!=1:raise ValueError('Row reuse geometry not prepared')
 c[FLAG]=direction;return c

def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);return prior.variant(c)+('|row_reuse_v1:'+c[FLAG] if c.get(FLAG) else '')
def family(c):
 c=normalize(c);return ('row_reuse_v1_'+c[FLAG]+'/' if c.get(FLAG) else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);return dict(schema='row_reuse_v31_1',direction=c.get(FLAG,'off'),computation_identity=computation_identity(c),
  compiler_variant=variant(c),shared_input=prior.descriptor(c),row_tile=32,row_pair=64,
  scope='single_kernel_adjacent_tiles_same_expert',shared_forward_precedence=True,dynamic_fallback='unchanged_B1_B3_long',across_updates_reuse=False)
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state

def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
 record=state['identity']
 if 'row_reuse_descriptor' in record:
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Row reuse resume descriptor mismatch')
  return 'VALIDATED_ROW_REUSE_V31'
 # Explicit legacy source identity: v30 same-policy checkpoint, no row-pair kernel.
 if not c.get(FLAG) and record.get('seal',{}).get('a100_speed_study_v30.py')=='74caa058b348935f9ccc3c6af9b0bda966fc310c64585d618060940727773f1c':
  return 'EXPLICIT_V30_ROW_REUSE_OFF_MIGRATION'
 raise ValueError('Missing row reuse descriptor or unaudited legacy source')
