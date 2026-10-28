"""User-authorized staged VRAM budgets; no CUDA work at import time."""
from dataclasses import dataclass, asdict
import math

GIB = 2**30

@dataclass(frozen=True)
class Tier:
    fraction: float
    reserve_gib: float
    physical_floor_gib: float

TIERS = {'baseline85': Tier(.85, 1.25, 3.),
         'expanded90': Tier(.90, .75, 2.),
         'expanded92': Tier(.92, .50, 1.5)}

def tier(name):
    return TIERS[name]

def describe(name):
    return dict(name=name, **asdict(tier(name)))

def allowance(name, total_gib, peaks, frees, maximum_gib=8.):
    """Additional tensor budget, not a promise of physical peak coverage."""
    assert peaks and frees
    values=[total_gib, maximum_gib, *peaks, *frees]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    t=tier(name)
    return max(0., min(maximum_gib, t.fraction*total_gib-max(peaks)-t.reserve_gib,
                       min(frees)-t.physical_floor_gib))

def validate(name, total_gib, allocated_peak_gib, observed_free_gib):
    t=tier(name)
    assert math.isfinite(allocated_peak_gib) and math.isfinite(observed_free_gib)
    assert allocated_peak_gib <= t.fraction*total_gib-t.reserve_gib, 'MEASURED_ALLOCATOR_RESERVE'
    assert observed_free_gib >= t.physical_floor_gib, 'MEASURED_PHYSICAL_RESERVE'

def growth_caps(budget_bytes):
    """Start at <=2 GiB, then add at most1 GiB between neighboring caps."""
    assert budget_bytes >= 0
    if not budget_bytes:return []
    return sorted({min(budget_bytes, n*GIB) for n in (2,3,4,5,6,7,8)})

def retention_projection(name,total_gib,rows,target):
    """A conservative admission estimate; actual gates are still mandatory."""
    data=sorted(rows,key=lambda x:x['retained'])
    if len(data)<2:return dict(admitted=False,reason='NEED_TWO_MEASURED_PARENTS')
    a,b=data[-2:]
    assert a['retained']<b['retained']<target
    slope=max(0.,(b['peak_gib']-a['peak_gib'])/(b['retained']-a['retained']))
    extra=max(.25,slope*(target-b['retained'])*1.25)
    peak=b['peak_gib']+extra;free=b['free_gib']-extra
    t=tier(name)
    return dict(admitted=peak <= t.fraction*total_gib-t.reserve_gib and free >= t.physical_floor_gib,
                projected_peak_gib=peak,projected_free_gib=free,extra_gib=extra,
                estimate_only=True,slope_gib_per_layer=slope)

class Admission:
    def __init__(self):
        self.active='baseline85';self.accepted=[];self.blocked=False
    def accept(self,name,*,numeric_pass,full_steps):
        order=list(TIERS)
        assert not self.blocked and name==order[len(self.accepted)]
        assert numeric_pass and full_steps>=32
        self.accepted.append(name);self.active=name
    def reject(self,name):
        assert name==list(TIERS)[len(self.accepted)]
        self.blocked=True

def reinvest_offload(cfg,row,numeric,total_gib):
    if cfg.get('memo_roles') or cfg.get('offload_bytes',0) not in (GIB,2*GIB):return []
    gates=[g for g in numeric if g['config']['name']==cfg['name']]
    peaks=[row['peak_gib']]+[g['peak_gib'] for g in gates]
    frees=[row.get('min_observed_free_gib',row['free_gib'])]+[g['free_gib'] for g in gates]
    cap=int(allowance(cfg['memory_tier'],total_gib,peaks,frees)*GIB)
    # Two bounded additions, based on a completed no-memo offload window.
    return [dict(cfg,name='reinvest_'+cfg['name']+'_'+str(n),memo_policy='all',
                 memo_roles=['gate_proj','up_proj','down_proj'],memo_bytes=n,
                 offload_parent_measured_peak_gib=max(peaks))
            for n in growth_caps(cap)[:2]]
