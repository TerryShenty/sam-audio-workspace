"""CPU-testable admission bookkeeping for a future isolated mixed-row worker.

The caller supplies only this allocation's completed numeric and training rows.
Scheduling eligibility is not candidate GPU acceptance: every descendant still
needs its own complete numeric/gradient/memory gate and 32-update window.
This component is not imported by the sealed v23 worker.
"""
import copy
import math

from memory_budget_v17 import validate
from pipeline_launch_candidate_v12 import launch_config
from row_tile_candidate_v23 import row_tile
from successor_design_v23 import computation_identity


def parent_key(config):
    return (config['retained'], config['static'], row_tile(config))


def supported(config):
    try:
        tile = row_tile(config)
        if tile not in (16, 32) or config.get('static') is not True:
            return False
        if config.get('lora_cache_bytes', 0) or config.get('offload_bytes', 0):
            return False
        for forward in (True, False):
            launch = launch_config(config, forward)
            if tile == 16 and launch['stages'] != 1:
                return False
        return True
    except (KeyError, ValueError, AssertionError):
        return False


class ParentAdmissions:
    def __init__(self):
        self._records = {}

    def register(self, config, row, numeric, *, memory_tier, total_gib):
        """Record one freshly completed parent; reject missing/mismatched proof."""
        assert supported(config), 'UNSUPPORTED_PARENT'
        key = parent_key(config)
        assert key not in self._records, 'PARENT_ALREADY_RECORDED'
        identity = computation_identity(config)
        assert computation_identity(row['config']) == identity, 'WINDOW_CONFIG_MISMATCH'
        assert row['steps'] == 32 and row['samples'] == 64, 'INCOMPLETE_WINDOW'
        assert math.isfinite(row['seconds_per_step']) and row['seconds_per_step'] > 0
        assert config['memory_tier'] == memory_tier
        assert row['config']['memory_tier'] == memory_tier
        cases = [g for g in numeric if computation_identity(g['config']) == identity]
        assert {g['case'] for g in cases} == {'mixed_sam', 'longest_qwen'}, 'INCOMPLETE_NUMERIC_CASES'
        for gate in cases:
            assert gate['config']['memory_tier'] == memory_tier
            assert len(gate['gradient_groups']) == 4, 'INCOMPLETE_GRADIENT_GROUPS'
            assert math.isfinite(gate['loss'])
            assert all(math.isfinite(v['cosine']) and math.isfinite(v['norm_ratio'])
                       and v['norm_ratio'] > 0 for v in gate['gradient_groups'].values())
        peak = max([row['peak_gib']] + [g['peak_gib'] for g in cases])
        free = min([row['min_observed_free_gib']] + [g['free_gib'] for g in cases])
        assert all(math.isfinite(v) and v >= 0 for v in (total_gib, peak, free))
        validate(memory_tier, total_gib, peak, free)
        self._records[key] = dict(config=copy.deepcopy(config), memo_bytes=config.get('memo_bytes', 0),
                                  peak_gib=peak, free_gib=free, memory_tier=memory_tier,
                                  full_steps=32, samples=64)

    def can_schedule(self, config):
        if not supported(config):
            return False
        record = self._records.get(parent_key(config))
        return (record is not None and config.get('memory_tier') == record['memory_tier']
                and 0 <= config.get('memo_bytes', 0) <= record['memo_bytes'])

    def records(self):
        return [dict(parent_key=list(k), **copy.deepcopy(v)) for k, v in self._records.items()]
