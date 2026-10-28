"""Fresh-gated projection descendants and bounded interaction batches."""
import copy
import itertools
from projection_identity_candidate_v25 import normalize, computation_identity, variant, required_seconds
from projection_launch_candidate_v25 import proposals as projection_singles, ROLES, SIDES
from successor_design_v24 import proposals as prior_proposals, family as prior_family
from successor_design_v23 import expected_regions, select_confirmed, oom_record
from memory_budget_v17 import describe
from parent_admission_v24_candidate import ParentAdmissions as PriorAdmissions, parent_key


class ParentAdmissions(PriorAdmissions):
    def register(self, config, row, numeric, **kwargs):
        config = normalize(config)
        row = dict(row, config=normalize(row['config']))
        numeric = [dict(g, config=normalize(g['config'])) for g in numeric]
        return super().register(config, row, numeric, **kwargs)

    def can_schedule(self, config):
        try:
            return super().can_schedule(normalize(config))
        except (KeyError, ValueError, AssertionError):
            return False


def family(config):
    return 'projection_row32' if normalize(config).get('projection_kernels') else prior_family(config)


def resolve_confirmed_chain(results):
    assert results, 'MISSING_CONFIRMED_SOURCE'
    result = results[0]
    assert result['status'] == 'PASS_SESSION_BUDGET_USED_WITH_MEASURED_CONTINUATION'
    selected = select_confirmed(result.get('summaries', {}))
    if selected: return copy.deepcopy(selected)
    assert result.get('inherited_confirmed_fallback') is True
    selected = resolve_confirmed_chain(results[1:])
    assert computation_identity(result['accepted_config']) == computation_identity(selected['config'])
    return selected


def eligible_prior(config, study, tier, total_gib):
    key = computation_identity(config)
    rows = [r for r in study['windows'] + study['numeric'] if computation_identity(r['config']) == key]
    bound = describe(tier)
    return bool(rows) and max(r['peak_gib'] for r in rows) <= total_gib*bound['fraction']-bound['reserve_gib']


def saturated_by(config, rows):
    for row in rows:
        old = row['config']
        if not old.get('memo_bytes') or config.get('memo_bytes',0) <= old['memo_bytes']: continue
        if row.get('memo_forward_calls',0) <= 0 or row.get('memo_replay_hits') != row['memo_forward_calls'] or row.get('memo',{}).get('live_bytes') != 0: continue
        a,b = dict(config),dict(old); a.pop('memo_bytes',None); b.pop('memo_bytes',None)
        if computation_identity(a) == computation_identity(b):
            return dict(reason='SAME_COMPUTATION_CACHE_ALREADY_SATURATED', measured_config=old)
    return None


def proposals(incumbent, extended=False):
    root = normalize(incumbent)
    # Cross-row roots still require their own fresh parent ledger entry.
    root.pop('nf4_row_tile',None); root.pop('projection_kernels',None)
    root['retained'] = min(root['retained'],24)
    for c in projection_singles(root): yield c
    projection_pool = []
    for retained in (20,16):
        if retained <= root['retained']:
            projection_pool.extend(projection_singles(dict(root,retained=retained)))
    if extended:
        for mode,native in (('both_dynamic_gemm','both'),('both_dynamic','native_rms'),('both_dynamic_gemm','contiguous')):
            projection_pool.extend(projection_singles(dict(root,compiler_mode=mode,native_sam_mode=native)))
        singles = list(projection_singles(root))
        for left,right in itertools.combinations(singles,2):
            c = copy.deepcopy(left)
            for role,sides in right['projection_kernels'].items():
                for side,values in sides.items():
                    c.setdefault('projection_kernels',{}).setdefault(role,{}).setdefault(side,{}).update(values)
            projection_pool.append(c)
        for role in ROLES:
            for side in SIDES:
                for values in (dict(group=2,waves=4),dict(group=4,waves=8),dict(bn=32,bk=64)):
                    c=copy.deepcopy(root);c['projection_kernels']={role:{side:values}}
                    projection_pool.append(c)
    others = [c for _,c in prior_proposals(incumbent,extended=extended)]
    for i in range(max(len(projection_pool),len(others))):
        if i < len(projection_pool): yield projection_pool[i]
        if i < len(others): yield others[i]


def generate(incumbent, admissions, evidence, planned, rejected, disabled, *, limit=24, extended=False):
    assert type(limit) is int and limit>0
    excluded={computation_identity(incumbent)}
    excluded.update(computation_identity(c) for c in planned)
    excluded.update(computation_identity(r['config']) for r in evidence+rejected if r.get('config'))
    result=[]
    for raw in proposals(incumbent,extended):
        c=normalize(raw); key=computation_identity(c)
        if key in excluded or variant(c) in disabled or not admissions.can_schedule(c) or saturated_by(c,evidence): continue
        c['name']=('adaptive' if extended else 'focused')+'_v25_'+str(len(planned)+len(result))
        result.append(c); excluded.add(key)
        if len(result)==limit: break
    return result


def finalists(rows, incumbent, minimum_gain=.015, limit=2):
    controls=[r['seconds_per_step'] for r in rows if r['phase'] in ('incumbent_recheck','time_control')]
    if not controls: return []
    reference=sum(controls[-3:])/len(controls[-3:]); selected=[]; seen={computation_identity(incumbent)}
    for row in sorted(rows,key=lambda r:r['seconds_per_step']):
        if row['phase'] not in ('factorial_screen','compiler_screen','refinement_screen','combination_screen'): continue
        config=row['config']; key=computation_identity(config)
        if key in seen or row['seconds_per_step']>reference*(1-minimum_gain): continue
        seen.add(key); selected.append(config)
        if len(selected)==limit: break
    return selected
