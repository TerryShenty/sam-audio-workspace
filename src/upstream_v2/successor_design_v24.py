"""Mixed-row successor queue; every output still needs fresh GPU gates."""
import collections
import copy

from parent_admission_v24_candidate import parent_key
from successor_design_v23 import (normalize, computation_identity, variant,
                                 required_seconds, saturated_by, select_confirmed)


def resolve_confirmed_chain(results):
    """Newest-to-oldest results, with exact identity at every fallback hop."""
    assert results, 'MISSING_CONFIRMED_SOURCE'
    result=results[0]
    assert result['status']=='PASS_SESSION_BUDGET_USED_WITH_MEASURED_CONTINUATION'
    selected=select_confirmed(result.get('summaries',{}))
    if selected:return copy.deepcopy(selected)
    assert result.get('inherited_confirmed_fallback') is True
    selected=resolve_confirmed_chain(results[1:])
    assert computation_identity(result['accepted_config'])==computation_identity(selected['config'])
    return selected


def family(config):
    if parent_key(config)[2]==16:return 'row_tile_16'
    if any(config.get(k,{}).get('stages',1)>1 for k in ('forward_kernel','backward_kernel')):
        return 'pipeline_row32'
    return 'confirmed_stack_row32'


def proposals(incumbent, *, extended=False):
    """Interleave demonstrated-stack descendants and changed weak interactions."""
    root=normalize(incumbent)
    for key in ('factors','lora_cache_bytes','offload_bytes','nf4_row_tile'):
        root.pop(key,None)
    # Keep unmeasured cross-row capacity at or below the measured incumbent.
    root['retained']=min(root['retained'],24)
    pools=collections.OrderedDict((n,[]) for n in ('geometry','compiler','retention','row16','pipeline'))

    def add(pool,**updates):
        c=copy.deepcopy(root);c.update(updates)
        if c.get('nf4_row_tile',32)==16:
            for side in ('forward_kernel','backward_kernel'):
                c[side]=dict(c[side],stages=1)
        pools[pool].append(c)

    for fw,bw in ((4,8),(4,4),(2,8),(8,4)):
        for fg,bg in ((2,4),(4,2),(4,4),(2,2)):
            add('geometry',forward_kernel=dict(root['forward_kernel'],waves=fw,group=fg),
                backward_kernel=dict(root['backward_kernel'],waves=bw,group=bg))
    for mode,native in (('both_dynamic','native_rms'),('both_dynamic','both'),
                        ('both_dynamic_gemm','both'),('both_dynamic_gemm','contiguous'),
                        ('qwen_attention_dynamic','native_rms')):
        for bg in (2,4):
            add('compiler',compiler_mode=mode,native_sam_mode=native,
                backward_kernel=dict(root['backward_kernel'],group=bg))
    for retained in (20,16):
        for bg in (1,2,4):
            add('retention',retained=retained,backward_kernel=dict(root['backward_kernel'],group=bg))
    for side in ('forward_kernel','backward_kernel'):
        for change in (dict(bk=64),dict(warps=8)):
            for group in (1,2):
                add('row16',nf4_row_tile=16,**{side:dict(root[side],**change,group=group)})
        for stages in (2,3):
            for group in (2,4):
                add('pipeline',**{side:dict(root[side],stages=stages,group=group)})
    if extended:
        # Further distinct interactions are queued in bounded batches; no
        # standalone speed cutoff permanently removes a method.
        for mode in ('both_dynamic_gemm','both_dynamic','qwen_attention_dynamic'):
            for native in ('native_rms','both','contiguous'):
                for fw,bw in ((2,8),(8,4),(4,4)):
                    add('compiler',compiler_mode=mode,native_sam_mode=native,
                        forward_kernel=dict(root['forward_kernel'],waves=fw,group=2),
                        backward_kernel=dict(root['backward_kernel'],waves=bw,group=4))
        for bn in (64,128):
            for fw,bw in ((2,4),(4,8),(8,2),(8,8)):
                for fg,bg in ((2,4),(4,2)):
                    add('geometry',forward_kernel=dict(root['forward_kernel'],bn=bn,waves=fw,group=fg),
                        backward_kernel=dict(root['backward_kernel'],waves=bw,group=bg))
        for retained in (20,16):
            for mode in ('both_dynamic_gemm','both_dynamic','qwen_attention_dynamic'):
                for native in ('native_rms','both'):
                    add('retention',retained=retained,compiler_mode=mode,native_sam_mode=native,
                        backward_kernel=dict(root['backward_kernel'],group=2))
        for side in ('forward_kernel','backward_kernel'):
            for change in (dict(bk=64),dict(warps=8)):
                for waves in (2,4):
                    add('row16',nf4_row_tile=16,**{side:dict(root[side],**change,waves=waves,group=4)})
            for stages in (2,3):
                for waves in (2,4):
                    add('pipeline',**{side:dict(root[side],stages=stages,waves=waves,group=2)})
    queues={k:collections.deque(v) for k,v in pools.items()}
    order=('geometry','compiler','row16','geometry','retention','pipeline')
    while any(queues.values()):
        for name in order:
            if queues[name]:yield name,queues[name].popleft()


def generate(incumbent,admissions,evidence,planned,rejected,disabled,*,limit=24,extended=False):
    assert isinstance(limit,int) and limit>0
    excluded={computation_identity(incumbent)}
    excluded.update(computation_identity(c) for c in planned)
    excluded.update(computation_identity(r['config']) for r in evidence+rejected if r.get('config'))
    result=[]
    for _,c in proposals(incumbent,extended=extended):
        key=computation_identity(c)
        if key in excluded or variant(c) in disabled or not admissions.can_schedule(c) or saturated_by(c,evidence):
            continue
        c['name']=('adaptive' if extended else 'focused')+'_v24_'+str(len(planned)+len(result))
        result.append(c);excluded.add(key)
        if len(result)==limit:break
    return result
