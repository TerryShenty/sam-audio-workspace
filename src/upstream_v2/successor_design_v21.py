"""Fresh-gated LoRA cache and NF4 geometry descendants of the confirmed stack."""
import copy
import math
from adaptive_design_v19 import canonical as old_canonical
from focused_design_v20 import expected_regions
from memory_budget_v17 import describe
from successor_design_v20 import select_confirmed, oom_record

GIB = 2**30


def normalize(cfg):
    c = copy.deepcopy(cfg)
    if not c.get('lora_cache_bytes', 0):
        c.pop('lora_cache_bytes', None)
    return c


def computation_identity(cfg):
    return old_canonical(normalize(cfg))


def variant(cfg):
    return (cfg.get('compiler_mode', 'existing') + '|' +
            cfg.get('native_sam_mode', 'existing') + '|' +
            ('cached' if cfg.get('lora_cache_bytes', 0) else 'plain'))


def required_seconds(cfg, rows):
    return 240 if any(variant(r['config']) == variant(cfg) and r.get('steps', 0) >= 32
                      for r in rows) else 900


def eligible_prior(cfg, study, tier, total_gib):
    rows = [r for r in study['windows'] + study['numeric']
            if computation_identity(r['config']) == computation_identity(cfg)]
    bound = describe(tier)
    return bool(rows) and max(r['peak_gib'] for r in rows) <= total_gib * bound['fraction'] - bound['reserve_gib']


def saturated_by(cfg, rows):
    for row in rows:
        old = row['config']
        for cap, complete in (
            ('memo_bytes', row.get('memo_forward_calls', 0) > 0 and
             row.get('memo_replay_hits') == row.get('memo_forward_calls') and
             row.get('memo', {}).get('live_bytes') == 0),
            ('lora_cache_bytes', row.get('lora_saved_matmuls', 0) > 0 and
             row.get('lora_replayed_calls', 0) > 0 and row.get('lora_budget_skips') == 0 and
             row.get('lora_cache', {}).get('live_bytes') == 0)):
            if not complete or cfg.get(cap, 0) <= old.get(cap, 0) or not old.get(cap, 0):
                continue
            left, right = dict(cfg), dict(old)
            left.pop(cap, None); right.pop(cap, None)
            if computation_identity(left) == computation_identity(right):
                return dict(reason='SAME_COMPUTATION_CACHE_ALREADY_SATURATED',
                            cache=cap, measured_config=old,
                            measured_seconds_per_step=row['seconds_per_step'])
    return None


def geometry_candidates(root):
    # Same existing Triton arithmetic, new matrix tile/warp choices on the
    # measured complete stack. No unbounded stage/autotune grid.
    defaults = dict(waves=root.get('waves', 4), group=root.get('group', 1),
                    bn=64, bk=32, warps=4, stages=1)
    shapes = [dict(bn=32), dict(bn=128), dict(bk=64), dict(warps=8),
              dict(bn=32, bk=64), dict(bn=128, bk=64),
              dict(bn=128, warps=8), dict(bn=64, bk=64, warps=8)]
    for changes in shapes:
        for side in ('forward_kernel', 'backward_kernel'):
            c = copy.deepcopy(root)
            c[side] = dict(defaults, **root.get(side, {}))
            c[side].update(changes)
            yield c
    for forward, backward in ((32,128), (128,32), (128,128), (32,32)):
        for waves in (2,4,8):
            c = copy.deepcopy(root)
            c['forward_kernel'] = dict(defaults, **root.get('forward_kernel', {}))
            c['backward_kernel'] = dict(defaults, **root.get('backward_kernel', {}))
            c['forward_kernel'].update(bn=forward, waves=waves)
            c['backward_kernel'].update(bn=backward)
            yield c


def allowed(c, parents, budgets):
    key = (c['retained'], c['static'])
    return (key in parents and c.get('memo_bytes', 0) <= budgets.get(key, 0)
            and c.get('lora_cache_bytes', 0) <= budgets.get('lora_extra_cap', 0))


def configurations(base, parents, budgets, incumbent, previous):
    root = normalize(incumbent)
    root.pop('lora_cache_bytes', None)
    pending = []
    caps = [n for n in (GIB//4, GIB//2) if n <= budgets.get('lora_extra_cap', 0)]
    # First establish the incremental cache on the exact four-confirmed stack.
    pending += [dict(root, lora_cache_bytes=n) for n in caps]
    kernels = list(geometry_candidates(root))
    cache_interactions = []
    for native in ('contiguous', 'both', 'existing'):
        for cap in caps:
            cache_interactions.append(dict(root, native_sam_mode=native, lora_cache_bytes=cap))
    for i in range(max(len(kernels), len(cache_interactions))):
        if i < len(kernels): pending.append(kernels[i])
        if i < len(cache_interactions): pending.append(cache_interactions[i])
    failed = {computation_identity(r['config']) for r in previous.get('rejected', []) if r.get('config')}
    seen = {computation_identity(root)} | failed
    result = []
    for c in pending:
        key = computation_identity(c)
        if key in seen or not allowed(c, parents, budgets): continue
        seen.add(key); c['name'] = 'focused_v21_' + str(len(result)); result.append(c)
    return result


def adaptive_configurations(incumbent, parents, budgets, rows, prior_rows,
                            planned, rejected, disabled, limit=24):
    evidence = prior_rows + rows
    excluded = {computation_identity(c) for c in planned}
    excluded.update(computation_identity(r['config']) for r in evidence + rejected if r.get('config'))
    roots = [normalize(incumbent)]
    seen_variants = {(incumbent.get('lora_cache_bytes',0), incumbent.get('native_sam_mode','existing'))}
    for r in sorted(rows, key=lambda x:x.get('seconds_per_step', math.inf)):
        c = r['config']; key = (c.get('lora_cache_bytes',0), c.get('native_sam_mode','existing'))
        if r.get('steps',0) < 32 or key in seen_variants or not allowed(c, parents, budgets): continue
        if not math.isfinite(r.get('seconds_per_step',math.inf)): continue
        roots.append(normalize(c)); seen_variants.add(key)
        if len(roots) >= 4: break
    pools = []
    for root in roots:
        pool = list(geometry_candidates(root))
        for mode in ('both_dynamic_gemm', 'both_dynamic', 'qwen_attention_dynamic'):
            for native in ('native_rms', 'contiguous', 'both', 'existing'):
                pool.append(dict(root, compiler_mode=mode, native_sam_mode=native))
        for fg,bg in ((2,1),(1,2),(4,1),(1,4),(2,2),(4,4)):
            for fw,bw in ((4,4),(2,4),(4,8),(2,8),(8,2)):
                c = copy.deepcopy(root)
                c['forward_kernel'] = dict(waves=fw,group=fg,bn=64,bk=32,warps=4,stages=1)
                c['backward_kernel'] = dict(waves=bw,group=bg,bn=64,bk=32,warps=4,stages=1)
                pool.append(c)
        pools.append(pool)
    result = []
    for i in range(max(map(len,pools), default=0)):
        for pool in pools:
            if i >= len(pool): continue
            c = pool[i]; key = computation_identity(c)
            if key in excluded or not allowed(c,parents,budgets) or variant(c) in disabled or saturated_by(c,evidence): continue
            c['name'] = 'adaptive_v21_' + str(len(planned)+len(result))
            result.append(c); excluded.add(key)
            if len(result) >= limit: return result
    return result
