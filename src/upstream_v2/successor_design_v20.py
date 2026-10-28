"""Confirmed successor selection, tier compatibility and diagnostic records."""
import copy,math,time
from adaptive_design_v19 import canonical as computation_identity,required_seconds
from adaptive_design_v19 import configurations as adaptive_configurations
from focused_design_v20 import configurations,expected_regions,saturated_by
from memory_budget_v17 import describe

def select_confirmed(summaries):
    eligible=[]
    for v in summaries.values():
        times=v.get('windows',[])
        if v.get('repeats',0)<4 or len(times)<4 or v.get('steps',0)<128:continue
        if not all(math.isfinite(t) and t>0 for t in times):continue
        if not math.isfinite(v.get('mean',float('nan'))):continue
        if not math.isclose(v['mean'],sum(times)/len(times),rel_tol=1e-9):continue
        eligible.append(v)
    return copy.deepcopy(min(eligible,key=lambda v:v['mean'])) if eligible else None

def eligible_prior(cfg,study,tier,total_gib):
    rows=[r for r in study['windows']+study['numeric']
          if computation_identity(r['config'])==computation_identity(cfg)]
    if not rows:return False
    bound=describe(tier)
    return max(r['peak_gib'] for r in rows)<=total_gib*bound['fraction']-bound['reserve_gib']

def oom_record(cfg,exc,phase,cuda,device,context=None):
    # Inspect before releasing traceback, zero_grad or empty_cache. Each query
    # can fail independently under a CUDA error; diagnostics must not mask it.
    r=dict(config=copy.deepcopy(cfg),reason='OOM',detail=str(exc),
        exception_type=type(exc).__name__,time=time.time(),phase=phase,
        operation_context=copy.deepcopy(context or {}),memory_before_cleanup={},
        allocator_fraction=describe(cfg['memory_tier'])['fraction'])
    for name,fn in [('allocated_bytes',lambda:cuda.memory_allocated(device)),
                    ('reserved_bytes',lambda:cuda.memory_reserved(device)),
                    ('peak_allocated_bytes',lambda:cuda.max_memory_allocated(device)),
                    ('peak_reserved_bytes',lambda:cuda.max_memory_reserved(device)),
                    ('physical_free_total_bytes',lambda:cuda.mem_get_info(device)),
                    ('allocator_stats',lambda:cuda.memory_stats(device))]:
        try:r['memory_before_cleanup'][name]=fn()
        except Exception as error:r['memory_before_cleanup'][name]={'query_error':repr(error)}
    return r
