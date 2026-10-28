"""Fresh row-tile descendants, preserving confirmed-state and memory contracts."""
import copy,math
from successor_design_v21 import (normalize as old_normalize,computation_identity as old_identity,
    expected_regions,select_confirmed,oom_record,variant as compiler_variant)
from memory_budget_v17 import describe
from row_tile_candidate_v23 import row_tile
GIB=2**30

def normalize(cfg):
    c=old_normalize(cfg)
    if row_tile(c)==32:c.pop('nf4_row_tile',None)
    return c

def computation_identity(cfg):return old_identity(normalize(cfg))
def variant(cfg):return compiler_variant(cfg)+'|row'+str(row_tile(cfg))
def required_seconds(cfg,rows):
    return 240 if any(variant(r['config'])==variant(cfg) and r.get('steps',0)>=32 for r in rows) else 900

def resolve_confirmed(previous,proven):
    assert previous['status']=='PASS_SESSION_BUDGET_USED_WITH_MEASURED_CONTINUATION'
    chosen=select_confirmed(previous.get('summaries',{}))
    if chosen:return chosen
    assert previous.get('inherited_confirmed_fallback') is True
    assert proven['status']=='PASS_SESSION_BUDGET_USED_WITH_MEASURED_CONTINUATION'
    chosen=select_confirmed(proven.get('summaries',{}));assert chosen
    assert computation_identity(previous['accepted_config'])==computation_identity(chosen['config'])
    return copy.deepcopy(chosen)

def eligible_prior(cfg,study,tier,total_gib):
    values=[r for r in study['windows']+study['numeric'] if computation_identity(r['config'])==computation_identity(cfg)]
    b=describe(tier)
    return bool(values) and max(r['peak_gib'] for r in values)<=total_gib*b['fraction']-b['reserve_gib']

def saturated_by(cfg,rows):
    for r in rows:
        old=r['config']
        if not old.get('memo_bytes') or cfg.get('memo_bytes',0)<=old['memo_bytes']:continue
        if r.get('memo_forward_calls',0)<=0 or r.get('memo_replay_hits')!=r['memo_forward_calls'] or r.get('memo',{}).get('live_bytes')!=0:continue
        a,b=dict(cfg),dict(old);a.pop('memo_bytes',None);b.pop('memo_bytes',None)
        if computation_identity(a)==computation_identity(b):return dict(reason='SAME_COMPUTATION_CACHE_ALREADY_SATURATED',measured_config=old)
    return None

def allowed(c,parents,budgets):
    key=(c['retained'],c['static'])
    if row_tile(c)!=16 or c.get('lora_cache_bytes',0) or key not in parents:return False
    if c.get('memo_bytes',0)>budgets.get(key,0):return False
    return all(c.get(k,{}).get('stages',1)==1 for k in ('forward_kernel','backward_kernel'))

def kernel(c,side,**kwargs):
    defaults=dict(waves=c.get('waves',4),group=c.get('group',1),bn=64,bk=32,warps=4,stages=1)
    c[side]=dict(defaults,**c.get(side,{}));c[side].update(kwargs)
    return c

def pools(incumbent,parents,budgets):
    roots=[dict(normalize(incumbent),nf4_row_tile=16)]
    for key,p in parents.items():
        if key[0]!=incumbent['retained'] and row_tile(p)==16:roots.append(normalize(p))
    geometry=[];compilers=[];groups=[]
    for root in roots:
        geometry.append(copy.deepcopy(root))
        for fw,bw in ((4,4),(2,8),(8,8),(4,2),(2,4),(8,4),(2,2),(8,2),(4,8)):
            for fn,bn in ((64,128),(128,128),(64,64),(128,64)):
                c=copy.deepcopy(root);kernel(c,'forward_kernel',waves=fw,bn=fn);kernel(c,'backward_kernel',waves=bw,bn=bn);geometry.append(c)
        for mode in ('both_dynamic_gemm','both_dynamic','qwen_attention_dynamic'):
            for native in ('native_rms','both','contiguous','existing'):
                compilers.append(dict(copy.deepcopy(root),compiler_mode=mode,native_sam_mode=native))
        for fg,bg in ((1,2),(2,1),(2,2),(1,4),(4,1)):
            c=copy.deepcopy(root);kernel(c,'forward_kernel',group=fg);kernel(c,'backward_kernel',group=bg);groups.append(c)
    def across_roots(values):
        per_root=len(values)//len(roots)
        return [values[r*per_root+i] for i in range(per_root) for r in range(len(roots))]
    return tuple(across_roots(values) for values in (geometry,compilers,groups))

def generate(incumbent,parents,budgets,evidence,planned,rejected,disabled,limit):
    excluded={computation_identity(c) for c in planned}
    excluded.update(computation_identity(r['config']) for r in evidence+rejected if r.get('config'))
    queues=pools(incumbent,parents,budgets);result=[]
    for i in range(max(map(len,queues),default=0)):
        for queue in queues:
            if i>=len(queue):continue
            c=queue[i];key=computation_identity(c)
            if key in excluded or variant(c) in disabled or not allowed(c,parents,budgets) or saturated_by(c,evidence):continue
            c['name']='focused_v23_'+str(len(planned)+len(result));result.append(c);excluded.add(key)
            if len(result)>=limit:return result
    return result

def configurations(base,parents,budgets,incumbent,previous):
    return generate(incumbent,parents,budgets,[],[],previous.get('rejected',[]),set(),36)

def adaptive_configurations(incumbent,parents,budgets,rows,prior_rows,planned,rejected,disabled,limit=24):
    out=generate(incumbent,parents,budgets,prior_rows+rows,planned,rejected,disabled,limit)
    for i,c in enumerate(out):c['name']='adaptive_v23_'+str(len(planned)+i)
    return out
