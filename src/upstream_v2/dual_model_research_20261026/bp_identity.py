"""Paired-byte NF4 decode over unchanged row32 routing and backward."""
import copy,json
from pathlib import Path
import const_codebook_identity_v60 as prior
import gate_up_identity_candidate_v28 as shared
from const_codebook_identity_v60 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,SCALE,GUARD,POINTWISE,AGGREGATE,PREP,INPUT,CODEBOOK,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped,has_scale,fused_scale_sides
BYTEPAIR='nf4_bytepair_decode_v1'
WORD4='nf4_word4_decode_v1'
WORD8='nf4_word8_decode_v1'
BOOKGATHER='nf4_bookgather_decode_v1'
BACKWARD='qwen_nf4_attention_backward_int8_30b_v1'
ATTENTION='qwen_nf4_attention_forward_int8_30b_v1'
WINNER=copy.deepcopy(prior.WINNER);WINNER[BYTEPAIR]=True;WINNER[WORD4]=True;WINNER[ATTENTION]=True;WINNER[WORD8]=True
def normalize(config):
 raw=dict(config);backward=raw.pop(BACKWARD,False)
 if type(backward) is not bool:raise ValueError('Explicit backward boolean required')
 gather=raw.pop(BOOKGATHER,False)
 if type(gather) is not bool:raise ValueError('Explicit bookgather boolean required')
 eight=raw.pop(WORD8,False)
 if type(eight) is not bool:raise ValueError('Explicit word8 boolean required')
 attention=raw.pop(ATTENTION,False)
 if type(attention) is not bool:raise ValueError('Explicit attention boolean required')
 value=raw.pop(BYTEPAIR,False);back=raw.pop(WORD4,False)
 if type(back) is not bool:raise ValueError("Explicit word4 forward boolean required")
 if type(value) is not bool:raise ValueError('Explicit paired-byte boolean required')
 if raw.pop('nf4_backward_scale_broadcast_v1',False) or raw.pop('nf4_splitk2_v1',False) or raw.pop('nf4_pairlut_v1',False) or raw.pop('nf4_backward_bytepair_v1',False) or raw.pop('nf4_blockint8_v1',False) or raw.pop('nf4_up_silu_epilogue_v1',False) or raw.pop('nf4_same_expert64_v1',False) or raw.pop('sam_wo_int8',False) or raw.pop('nf4_gate_up_scaled_rowpair_v1',False):raise ValueError('Closed scaled gate/up candidate')
 c=prior.normalize(raw)
 if c.get(CODEBOOK) or c.get('nf4_gate_up_shared_input'):raise ValueError('Closed codebook/shared-input candidates')
 if value:
  if prior.computation_identity(c)!=prior.computation_identity(prior.WINNER):raise ValueError('Only the fixed accepted winner')
  c[BYTEPAIR]=True
 if back:
  if not value:raise ValueError("Accepted forward bytepair must remain enabled")
  c[WORD4]=True
 if attention:
  if not (value and back):raise ValueError('Attention requires accepted V3 word4 path')
  c[ATTENTION]=True
 if eight:
  if not(value and back and attention):raise ValueError('Word8 requires accepted V4 attention and word4')
  c[WORD8]=True
 if gather:raise ValueError('Closed bookgather candidate')
 if backward:
  if not(value and back and attention and eight):raise ValueError('Backward requires all V5 flags')
  c[BACKWARD]=True
 return c
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
 c=normalize(c);value=c.pop(BYTEPAIR,False)
 return prior.variant(c)+('|bytepair_v1' if value else '')+('|word4_decode_v1' if c.get(WORD4) else '')+('|word8_decode_v1' if c.get(WORD8) else '')+('|bookgather_decode_v1' if c.get(BOOKGATHER) else '')
def family(c):
 c=normalize(c);value=c.pop(BYTEPAIR,False)
 return ('bookgather_decode_v1/' if c.get(BOOKGATHER) else '')+('word8_decode_v1/' if c.get(WORD8) else '')+('word4_decode_v1/' if c.get(WORD4) else '')+('bytepair_v1/' if value else '')+prior.family(c)
def descriptor(c):
 c=normalize(c);d=prior.descriptor(c)
 d.update(schema='nf4_attention_backward_int8_30b_v1',bytepair=bool(c.get(BYTEPAIR)),compiler_variant=variant(c),bytepair_implementation='bytepair_launch',source_contract='bp_live',bytepair_geometry={'bm':32,'bn':64,'bk':32,'warps':4,'stages':1,'waves':32},bytepair_backward='unchanged original fusion',bytepair_quality_equivalence=False)
 d.update(word4_decode=bool(c.get(WORD4)),word4_decode_implementation='word4_launch',word4_decode_scale='unchanged BF16 cached forward scales',word4_decode_geometry={'bn':64,'bk':32,'warps':4,'stages':1,'waves':32})
 d.update(attention_forward_int8=bool(c.get(ATTENTION)),attention_backward_int8=bool(c.get(BACKWARD)),attention_backward='approximate_dynamic_INT8' if c.get(BACKWARD) else 'original_NF4_BF16',quality_equivalence=False)
 d.update(word8_decode=bool(c.get(WORD8)),word8_decode_implementation='word8_launch',word8_decode_geometry={'bn':64,'bk':32,'warps':4,'stages':1,'waves':32})
 d.update(bookgather_decode=bool(c.get(BOOKGATHER)),bookgather_decode_implementation='bookgather_launch',bookgather_decode_geometry={'bn':64,'bk':32,'warps':4,'stages':1,'waves':32})
 return d
def stamp_continuation(state,c):
 c=normalize(c);state=prior.stamp_continuation(state,c)
 state['identity']['row_reuse_descriptor']=descriptor(c);return state
def validate_resume_config(state,c,*,allow_audited_legacy=False):
 c=normalize(c);record=state['identity']
 if allow_audited_legacy:raise ValueError('Explicit latest source only')
 shared.validate_resume_config(state,c,allow_audited_legacy=False)
 if record.get('row_reuse_descriptor',{}).get('schema')=='nf4_attention_backward_int8_30b_v1':
  if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Paired-byte descriptor mismatch')
  return 'VALIDATED_BYTEPAIR'
 raise ValueError('Only this export current descriptor is accepted; research state migration is disabled')
