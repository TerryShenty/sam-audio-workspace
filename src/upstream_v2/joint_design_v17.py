"""Native SAM interactions with safe Qwen parents and unfinished v13 work."""
from exploration_design_v11 import interleave,identity
from memory_budget_v17 import growth_caps


def expected_regions(cfg):
    mode=cfg.get('compiler_mode','existing');regions=set()
    if mode in ('sam_dynamic','sam_autotune','both_dynamic','sam_dynamic_gemm','both_dynamic_gemm') or cfg.get('native_sam_mode','existing')!='existing':
        regions.add('sam')
    if mode in ('qwen_attention_dynamic','both_dynamic','both_dynamic_gemm'):
        regions.add('qwen_attention')
    return regions


def configurations(base,parents,budgets,incumbent,previous):
    def normalized(c):
        c=dict(c);c['memory_tier']=base['memory_tier'];c.setdefault('native_sam_mode','existing');c.setdefault('compiler_mode','existing')
        return c
    def safe(c):
        key=(c['retained'],c['static'])
        return key in parents and c.get('memo_bytes',0)<=budgets.get(key,0)
    # Current-session parent/winner windows are reused by the worker, not
    # counted as extra screens or historical confirmation repetitions.
    roots=[]
    parent=parents.get((0,True))
    if parent:roots.append(('parent',normalized(parent)))
    memo=normalized(incumbent)
    if safe(memo):roots.append(('memo',memo))
    native=[];regional=[];directional=[];memory=[]
    for mode in ('existing','sam_dynamic_gemm'):
        for native_mode in ('existing','both','contiguous','native_rms'):
            for label,root in roots:
                native.append(dict(root,name=f'native_{label}_{mode}_{native_mode}',
                    native_sam_mode=native_mode,compiler_mode=mode))
    for label,root in roots:
        root=dict(root,compiler_mode='existing',native_sam_mode='existing')
        for native_mode in ('existing','both'):
            regional.append(dict(root,name=f'native_joint_compile_{label}_{native_mode}',
                native_sam_mode=native_mode,compiler_mode='both_dynamic_gemm'))
        regional.append(dict(root,name=f'native_qwen_compile_{label}',
            native_sam_mode='both',compiler_mode='qwen_attention_dynamic'))
        if label=='memo':
            for direction in ('forward','backward','both'):
                for waves in (2,8):
                    c=dict(root,name=f'native_kernel_{direction}_w{waves}',native_sam_mode='both')
                    kernel=dict(waves=waves,group=1,bn=64,bk=32,warps=4,stages=1)
                    if direction in ('forward','both'):c['forward_kernel']=dict(kernel)
                    if direction in ('backward','both'):c['backward_kernel']=dict(kernel)
                    directional.append(c)
    # Spend measured headroom on retention + Qwen compilation, not only a
    # larger memo whose 6GiB configuration already nearly saturates replay hits.
    for retained in (0,8,16,24,32,40):
        parent=parents.get((retained,True));budget=budgets.get((retained,True),0)
        if parent is None or budget<=0:continue
        for cap in growth_caps(budget):
            memory.append(dict(normalized(parent),name=f'qwen_memory_r{retained}_{cap}',
                memo_policy='all',memo_roles=['gate_proj','up_proj','down_proj'],memo_bytes=cap,
                compiler_mode='qwen_attention_dynamic'))
    failed={identity(normalized(x['config'])) for x in previous.get('rejected',[])
            if x.get('config') and x.get('reason') not in ('PARENT_NOT_SAFE','MEMO_GATE_OR_MEMORY_BUDGET_UNAVAILABLE')}
    unfinished=[normalized(c) for c in previous.get('untested_candidates',[])
                if safe(c) and identity(normalized(c)) not in failed]
    return interleave(memory,native,regional,directional,unfinished)
