"""Focused descendants of the confirmed full-step winner; no broad weak grid."""
from exploration_design_v11 import identity
from memory_budget_v17 import growth_caps
from joint_design_v17 import expected_regions


def computation_identity(cfg):
    cfg=dict(cfg)
    cfg.pop('memory_tier',None)
    cfg.setdefault('compiler_mode','existing');cfg.setdefault('native_sam_mode','existing')
    if not cfg.get('memo_verify',False):cfg.pop('memo_verify',None)
    if not cfg.get('offload_bytes',0):cfg.pop('offload_bytes',None)
    if cfg.get('memo_layers')==48-cfg.get('retained',0):cfg.pop('memo_layers',None)
    return identity(cfg)


def cache_family(cfg):
    # Only capacity differs; a change in retention, roles, kernel, or compiler
    # is a new hypothesis and must not be rejected by another family's hits.
    return computation_identity({k:v for k,v in cfg.items() if k!='memo_bytes'})


def saturated_by(cfg,rows):
    for row in rows:
        old=row.get('config',{})
        if not old.get('memo_bytes') or cfg.get('memo_bytes',0)<=old['memo_bytes']:
            continue
        if cache_family(cfg)!=cache_family(old):continue
        calls=row.get('memo_forward_calls',0)
        if calls>0 and row.get('memo_replay_hits')==calls and row.get('memo',{}).get('live_bytes')==0:
            return dict(reason='SAME_COMPUTATION_ALL_REPLAYS_ALREADY_HIT',
                measured_config=old,measured_cap_bytes=old['memo_bytes'],hits=calls,
                measured_seconds_per_step=row['seconds_per_step'])
    return None


def configurations(base,parents,budgets,incumbent,previous):
    root=dict(incumbent,memory_tier=base['memory_tier'])
    root.pop('factors',None)
    # Preserve every accepted arithmetic optimization while changing one axis.
    memory=[];kernel=[];sam=[]
    for retained in (8,16,24,32,40,0):
        parent=parents.get((retained,True));budget=budgets.get((retained,True),0)
        if parent is None or budget<=0:continue
        for cap in growth_caps(budget):
            if retained==root['retained'] and cap<=root.get('memo_bytes',0):continue
            c=dict(root,name=f'focused_retention_r{retained}_memo{cap}',retained=retained,
                static=True,memo_bytes=cap)
            c.pop('memo_layers',None)
            memory.append(c)
    for direction in ('forward','backward','both'):
        for waves in (2,8):
            c=dict(root,name=f'focused_kernel_{direction}_w{waves}')
            for k in ('forward_kernel','backward_kernel'):c.pop(k,None)
            tile=dict(waves=waves,group=1,bn=64,bk=32,warps=4,stages=1)
            if direction in ('forward','both'):c['forward_kernel']=dict(tile)
            if direction in ('backward','both'):c['backward_kernel']=dict(tile)
            kernel.append(c)
    for mode in ('contiguous','native_rms','both'):
        if mode!=root.get('native_sam_mode','existing'):
            sam.append(dict(root,name='focused_sam_'+mode,native_sam_mode=mode))
    # Two capacity/recomputation hypotheses then one packed-NF4 hypothesis.
    # SAM receives at most three incremental tests, after the primary work.
    ordered=[]
    while memory or kernel:
        ordered.extend(memory[:2]);memory=memory[2:]
        ordered.extend(kernel[:1]);kernel=kernel[1:]
    ordered.extend(sam)
    def normalized(c):return dict(c,memory_tier=base['memory_tier'])
    failed={computation_identity(normalized(x['config'])) for x in previous.get('rejected',[])
        if x.get('config') and x.get('reason') not in ('PARENT_NOT_SAFE','MEMO_GATE_OR_MEMORY_BUDGET_UNAVAILABLE')}
    result=[];seen={computation_identity(root)}
    for c in ordered:
        key=computation_identity(c)
        if key in failed or key in seen:continue
        if (c['retained'],c['static']) not in parents:continue
        if c.get('memo_bytes',0)>budgets[(c['retained'],c['static'])]:continue
        seen.add(key);result.append(c)
    return result
