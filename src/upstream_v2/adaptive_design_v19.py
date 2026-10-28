"""Continue with new, bounded interactions after the focused first queue.

Only already implemented arithmetic paths are composed. GPU numerical and
memory gates remain mandatory; a generated configuration is not accepted.
"""
import copy
import math
from focused_design_v18 import computation_identity, saturated_by


def canonical(cfg):
    c=copy.deepcopy(cfg)
    defaults=dict(waves=c.get('waves',4),group=c.get('group',1),bn=64,bk=32,warps=4,stages=1)
    for side in ('forward_kernel','backward_kernel'):
        actual=dict(defaults,**c.get(side,{}))
        if actual==defaults:c.pop(side,None)
        elif side in c:c[side]=actual
    return computation_identity(c)


def variant(cfg):
    return cfg.get('compiler_mode','existing')+'|'+cfg.get('native_sam_mode','existing')


def required_seconds(cfg,rows):
    # A previously returned backend and completed window is evidence of a warm
    # mode, not a guarantee that every new shape avoids compilation.
    warm=any(variant(r['config'])==variant(cfg) and r.get('steps',0)>=32 for r in rows)
    return 240 if warm else 900


def configurations(incumbent,parents,budgets,rows,prior_rows,planned,rejected,disabled,limit=24):
    evidence=prior_rows+rows
    excluded={canonical(c) for c in planned}
    excluded.update(canonical(r['config']) for r in evidence)
    excluded.update(canonical(r['config']) for r in rejected if r.get('config'))
    roots=[copy.deepcopy(incumbent)]
    retained_seen={incumbent['retained']}
    # Favor actually fast measured capacity/recomputation choices, while
    # retaining cross-axis combinations even when one isolated knob was slow.
    for r in sorted(rows,key=lambda x:x.get('seconds_per_step',float('inf'))):
        c=r['config'];t=r.get('seconds_per_step',float('inf'))
        if not math.isfinite(t) or t<=0 or r.get('steps',0)<32:continue
        if not c.get('static') or not c.get('memo_bytes') or c.get('retained') in retained_seen:continue
        if c.get('compiler_mode') not in ('qwen_attention_dynamic','both_dynamic','both_dynamic_gemm'):continue
        roots.append(copy.deepcopy(c));retained_seen.add(c['retained'])
        if len(roots)>=4:break
    pools=[[],[],[]]
    for root in roots:
        key=(root['retained'],root['static'])
        if key not in parents or root.get('memo_bytes',0)>budgets.get(key,0):continue
        # Existing forward/backward geometries, now crossed independently on
        # measured retained+memo roots. No new tile widths or pipeline stages.
        for fw,bw in ((2,4),(4,2),(8,4),(4,8),(2,8),(8,2),(2,2),(8,8)):
            c=copy.deepcopy(root)
            c['forward_kernel']=dict(waves=fw,group=1,bn=64,bk=32,warps=4,stages=1)
            c['backward_kernel']=dict(waves=bw,group=1,bn=64,bk=32,warps=4,stages=1)
            pools[0].append(c)
        # Keep Qwen compilation and memo while checking SAM compiler/native
        # interactions, never the old standalone no-memo broad grid.
        for compiler in ('qwen_attention_dynamic','both_dynamic','both_dynamic_gemm'):
            for native in ('existing','contiguous','native_rms','both'):
                c=copy.deepcopy(root);c.update(compiler_mode=compiler,native_sam_mode=native)
                pools[1].append(c)
        # Routing launch order and asymmetric wave count cross-products are
        # later alternatives, preserving the full differentiable workload.
        for fg,bg in ((2,1),(1,2),(4,1),(1,4),(2,2),(4,4),(2,4),(4,2)):
            for fw,bw in ((4,4),(2,4),(4,8),(2,8),(8,2)):
                c=copy.deepcopy(root)
                c['forward_kernel']=dict(waves=fw,group=fg,bn=64,bk=32,warps=4,stages=1)
                c['backward_kernel']=dict(waves=bw,group=bg,bn=64,bk=32,warps=4,stages=1)
                pools[2].append(c)
    result=[]
    for i in range(max(map(len,pools),default=0)):
        for pool in pools:
            if i>=len(pool):continue
            c=pool[i];key=canonical(c)
            if key in excluded or variant(c) in disabled or saturated_by(c,evidence):continue
            c['name']='adaptive_'+str(len(planned)+len(result))
            excluded.add(key);result.append(c)
            if len(result)>=limit:return result
    return result
