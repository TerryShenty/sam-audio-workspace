"""Explicit frozen scale metadata identity and audited v43 source migration."""
import copy,hashlib,json
import row_grouped_identity_v41 as prior
import gate_up_identity_candidate_v28 as shared
from row_grouped_identity_v41 import ROLES,SIDES,FLAG,WAVES,TUNING,FUSION,GROUPED,geometry,has_bk64,has_extended_waves,has_tuning,has_fusion,has_grouped
SCALE='nf4_row_scale_cache'
V43_WORKER_SHA='3d6a159b872a4ac0aab44b1061bec7c1724c762ef8e80a0c41932fbc95ed74f8'
def normalize(config):
    c=prior.normalize(config);raw=c.pop(SCALE,None)
    if raw is None or raw=={}:return c
    if not isinstance(raw,dict) or set(raw)!={'precision','roles','direction','max_bytes'}:raise ValueError('Complete scale policy required')
    if raw['precision'] not in ('fp32','bf16') or raw['direction'] not in ('forward','backward','both'):raise ValueError('Unknown scale precision or direction')
    roles=raw['roles']
    if not isinstance(roles,(list,tuple)) or not roles or len(set(roles))!=len(roles) or set(roles)-set(ROLES):raise ValueError('Explicit distinct scale roles required')
    if type(raw['max_bytes']) is not int or not 0<raw['max_bytes']<=2*2**30:raise ValueError('Scale budget must be positive and at most 2GiB')
    if not c.get('static') or c.get('nf4_row_reuse_tiles',2)!=2 or c.get('nf4_gate_up_shared_input'):raise ValueError('Scale cache requires separate static paired path')
    for role in roles:
        for side in SIDES:
            direction=side.split('_')[0]
            if raw['direction'] not in ('both',direction):continue
            g=geometry(c,role,side=='forward_kernel')
            if c.get(FLAG) not in ('both',direction) or g['bn'] not in (64,128) or (g['bk'],g['warps'],g['stages'])!=(32,4,1) or c.get(FUSION,{}).get(role,{}).get(side) or c.get(GROUPED,{}).get(role,{}).get(side):raise ValueError('Scale path requires nonfused nongrouped BK32/warps4/stages1')
    c[SCALE]=dict(precision=raw['precision'],roles=[r for r in ROLES if r in roles],direction=raw['direction'],max_bytes=raw['max_bytes'])
    return c

def has_scale(c):return bool(normalize(c).get(SCALE))
def computation_identity(c):return prior.computation_identity(normalize(c))
def variant(c):
    c=normalize(c);base=prior.variant(c)
    if c.get(SCALE):base+='|scale_pair_v44:'+hashlib.sha256(json.dumps(c[SCALE],sort_keys=True).encode()).hexdigest()
    return base

def family(c):
    c=normalize(c);return ('scale_pair_v44_'+c[SCALE]['precision']+'/' if c.get(SCALE) else '')+prior.family(c)
def descriptor(c):
    c=normalize(c);d=prior.descriptor(c);d.update(schema='row_reuse_v44_1',compiler_variant=variant(c),scale_policy=copy.deepcopy(c.get(SCALE,{})),scale_scope='immutable_nf4_quantization_metadata_only',scale_rebuild_after_process_resume=True);return d

def stamp_continuation(state,c):
    c=normalize(c);state=prior.stamp_continuation(state,c);state['identity']['row_reuse_descriptor']=descriptor(c);return state

def validate_resume_config(state,c,*,allow_audited_legacy=False):
    c=normalize(c);record=state['identity']
    if record.get('row_reuse_descriptor',{}).get('schema')=='row_reuse_v44_1':
        shared.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
        if record['row_reuse_descriptor']!=descriptor(c):raise ValueError('Scale-pair descriptor mismatch')
        return 'VALIDATED_SCALE_PAIR_V44'
    if not c.get(SCALE) and record.get('seal',{}).get('a100_speed_study_v43.py')==V43_WORKER_SHA:
        prior.validate_resume_config(state,c,allow_audited_legacy=allow_audited_legacy)
        return 'EXPLICIT_V43_TO_SCALE_PAIR_V44'
    raise ValueError('Missing scale descriptor or unaudited source')
