"""Publish validation only after exact all-rank identity/seed/score coverage."""
import math
from collections import Counter
METRICS=('si_sdr_db','aligned_si_sdr_db','si_sdri_db','activity_primary_f1')
def collect(rank_results,protocol,*,world_size=4):
    assert type(world_size) is int and 1<=world_size<=8
    assert len(rank_results)==world_size and {x['rank'] for x in rank_results}==set(range(world_size))
    expected={x['sample_id']:x for x in protocol['rows']}
    assert len(expected)==len(protocol['rows'])==817
    identity={k:rank_results[0][k] for k in ('checkpoint_identity','validation_identity')}
    for value in identity.values():assert isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    assert all(all(result[k]==v for k,v in identity.items()) for result in rank_results),'MIXED_CHECKPOINT_OR_VALIDATION_SOURCE'
    seen={};by_dataset={} 
    for result in rank_results:
        assert result['complete'] is True
        assert result['weight_choice']=='live' and result['solver']=='midpoint' and result['steps']==16
        for row in result['rows']:
            sid=row['sample_id'];assert sid in expected and sid not in seen,'MISSING_UNKNOWN_OR_DUPLICATE_ID'
            source=expected[sid]
            assert row['noise_seed']==source['noise_seed'] and row['dataset']==source['dataset']
            assert row['row_index']==source['original_row_index']
            assert type(row['noise_sha256']) is str and len(row['noise_sha256'])==64
            assert all(c in '0123456789abcdef' for c in row['noise_sha256'])
            assert all(math.isfinite(row[k]) for k in METRICS)
            assert type(row['evaluation_samples']) is int and row['evaluation_samples']>0
            seen[sid]=row;by_dataset.setdefault(row['dataset'],[]).append(row)
    assert set(seen)==set(expected),'INCOMPLETE_817_VALIDATION'
    assert Counter(x['dataset'] for x in seen.values())==Counter(x['dataset'] for x in expected.values())
    def average(rows):return {k:math.fsum(x[k] for x in rows)/len(rows) for k in METRICS}
    return dict(status='PASS_COMPLETE_817_SCORE_COVERAGE',examples=817,counts={k:len(v) for k,v in by_dataset.items()},
                mean=average(list(seen.values())),by_dataset={k:average(v) for k,v in by_dataset.items()},
                rows=[seen[x['sample_id']] for x in protocol['rows']],weight_choice='live',solver='midpoint',steps=16,**identity)
