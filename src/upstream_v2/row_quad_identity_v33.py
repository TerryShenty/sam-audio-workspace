"""Row32 groups of two or four; explicit same-policy v32 migration."""
import row_reuse_identity_v31 as prior
FLAG=prior.FLAG
TILES='nf4_row_reuse_tiles'
def normalize(config):
 c=prior.normalize(config);n=c.get(TILES,2)
 if type(n) is not int or n not in (2,4):raise ValueError('Prepared row groups are two or four tiles')
 if n==4 and not c.get(FLAG):raise ValueError('Quad requires an active row-reuse direction')
 if n==2:c.pop(TILES,None)
 return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);return prior.variant(c)+('|row_quad_v1' if c.get(TILES)==4 else '')
def family(c):
 c=normalize(c);return ('row_quad_v1/' if c.get(TILES)==4 else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v33_1',row_pair=32*c.get(TILES,2),row_tiles=c.get(TILES,2),compiler_variant=variant(c),scope='single_kernel_adjacent_row32_tiles_same_expert')
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v33_1':
  prior.prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Row group resume descriptor mismatch')
  return 'VALIDATED_ROW_QUAD_V33'
 if c.get(TILES,2)==2 and record.get('seal',{}).get('a100_speed_study_v32.py')=='f701e69cbc7948b6c0d4027f6681fcd05022aa505229a169a35f7ce65298ef82':
  prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
  return 'EXPLICIT_V32_TWO_TILE_MIGRATION'
 raise ValueError('Missing row group descriptor or unaudited source')
