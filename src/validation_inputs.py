"""Exact held-out identities/seeds and bounded batches; never selects training rows."""
from collections import Counter
EXPECTED={'urmp':412,'choralsynth':120,'easycom':207,'msdwild':78}

def bind(rows,protocol):
    assert protocol['status']=='PASS_ALL_817_ORIGINAL_IDENTITY_AND_NOISE_SEED_BINDING'
    items=[r for split,values in sorted(rows.items()) if split.endswith('_validation') for r in values]
    lookup={r['sample_id']:r for r in items};specs=protocol['rows']
    assert len(items)==len(lookup)==len(specs)==817
    assert len({s['sample_id'] for s in specs})==817 and set(lookup)=={s['sample_id'] for s in specs}
    assert Counter(r['dataset'] for r in items)==EXPECTED
    result=[]
    for s in specs:
        item=lookup[s['sample_id']]
        assert item['split'].endswith('_validation') and item['dataset']==s['dataset']
        assert item['media_key']==s['media_key'] and item['prompt']==s['prompt']
        assert type(s['noise_seed']) is int and s['noise_seed']>=0
        result.append(dict(item=item,protocol=s))
    return result

def batches(items,*,rank,world_size=4,batch_size=8):
    assert world_size==4 and 0<=rank<4 and batch_size in (1,2,4,8)
    # Global identity ordering is fixed; changing inference batching never changes seeds/rank.
    local=[x for i,x in enumerate(items) if i%world_size==rank]
    return [local[i:i+batch_size] for i in range(0,len(local),batch_size)]

def condition_pairs(part):
    """B2 Qwen conditioning: duplicate only an odd final input, discard its output.

    The second value is the number of real outputs. It must control slicing;
    duplicated conditioning work is never a scored example or sampler update.
    """
    assert 1<=len(part)<=8
    return [(part[i:i+2] if i+1<len(part) else [part[i],part[i]],min(2,len(part)-i))
            for i in range(0,len(part),2)]
