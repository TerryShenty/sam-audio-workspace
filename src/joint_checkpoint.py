"""N-rank checkpoint transaction; callers own collective coordination.

Only publish COMPLETE after all durable rank files agree on step/identity.
Full-model effective restore and target GPU memory gates remain caller duties.
"""
import json, os
from pathlib import Path

FIELDS={'model','buffers','optimizer','scheduler','ema','rng','sampler',
        'identity','configuration','step','rank','groups'}
GROUPS={'qwen','bridge','sam_lora','sam_conditions'}

def validate_state(state):
    assert FIELDS<=state.keys()
    assert type(state['step']) is int and state['step']>=0
    world=state['configuration']['world_size'];assert type(world) is int and 1<=world<=8
    assert state['rank'] in range(world) and state['identity'] and state['configuration']
    assert set(state['groups'])==GROUPS and all(state['groups'].values())
    names=[n for group in state['groups'].values() for n in group]
    assert len(names)==len(set(names)) and set(names)==set(state['model'])==set(state['ema'])
    assert {'python','numpy','torch','cuda'}<=state['rng'].keys()
    sampler=state['sampler']
    assert sampler['rank']==state['rank'] and sampler['world_size']==world and sampler['per_rank']==2
    assert sampler['global_samples']==2*world*state['step']
    assert state['configuration']['per_rank_batch']==2 and state['configuration']['global_batch']==2*world

def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def save_rank(directory,state):
    import torch
    validate_state(state)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'COMPLETE.json').exists()
    final=directory/f"rank{state['rank']}.pt";temporary=final.with_suffix('.partial')
    assert not final.exists(),'Never overwrite a checkpoint rank'
    # Exclusive creation preserves unknown/failed attempts for diagnosis.
    with temporary.open('xb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    temporary.rename(final);_sync_dir(directory)
    stat=final.stat()
    return dict(rank=state['rank'],name=final.name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                step=state['step'],identity=state['identity'],configuration=state['configuration'],
                sampler_identity=state['sampler']['identity'],global_samples=state['sampler']['global_samples'])

def publish(directory,receipts):
    directory=Path(directory).resolve()
    assert receipts
    world=receipts[0]['configuration']['world_size'];assert type(world) is int and 1<=world<=8
    assert len(receipts)==world and {r['rank'] for r in receipts}==set(range(world))
    reference=receipts[0]
    for r in receipts:
        for key in ('step','identity','configuration','sampler_identity','global_samples'):
            assert r[key]==reference[key],('CHECKPOINT_RANK_DISAGREEMENT',key)
        assert r['name']==f"rank{r['rank']}.pt"
        p=directory/r['name'];assert p.resolve().parent==directory
        s=p.stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']) and s.st_size>0
    manifest=dict(complete=True,world_size=world,step=reference['step'],identity=reference['identity'],
                  configuration=reference['configuration'],ranks=sorted(receipts,key=lambda r:r['rank']),
                  effective_resume_verified=False)
    temporary=directory/'COMPLETE.partial';final=directory/'COMPLETE.json'
    assert not final.exists(),'Never replace a published checkpoint'
    with temporary.open('x') as f:
        json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.rename(final);_sync_dir(directory)
    return final

def load_rank(manifest_path,*,rank,identity,configuration):
    import torch
    path=Path(manifest_path).resolve();manifest=json.loads(path.read_text())
    world=configuration['world_size'];assert type(world) is int and 1<=world<=8
    assert manifest['complete'] is True and manifest['world_size']==world,'Resume requires the same GPU count'
    assert manifest['identity']==identity and manifest['configuration']==configuration
    receipts=manifest['ranks'];assert len(receipts)==world and {r['rank'] for r in receipts}==set(range(world))
    for r in receipts:
        assert r['name']==f"rank{r['rank']}.pt"
        assert (path.parent/r['name']).resolve().parent==path.parent
        s=(path.parent/r['name']).stat()
        assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns'])
    receipt=next(r for r in receipts if r['rank']==rank)
    # Trusted locally generated full states include Python/NumPy RNG objects.
    state=torch.load(path.parent/receipt['name'],map_location='cpu',weights_only=False)
    s=(path.parent/receipt['name']).stat()
    assert (s.st_size,s.st_mtime_ns)==(receipt['bytes'],receipt['mtime_ns'])
    validate_state(state)
    assert state['rank']==rank and state['step']==manifest['step']
    assert state['identity']==identity and state['configuration']==configuration
    return state
