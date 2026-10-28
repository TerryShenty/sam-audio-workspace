"""Target-only full restore gate. Caller owns sealed source/input/memory admission."""
import json,time,hashlib
from pathlib import Path

def restore_gate(session,directory,*,max_tokens,long_input_receipt):
    import torch
    import torch.distributed as dist
    from qualification_observer import QualificationObserver
    from qualification_restore import restore_verified
    from deployment_guard import actual_activation,full_restore,BASELINE_SHA
    assert dist.is_initialized() and 1<=dist.get_world_size()<=8 and dist.get_backend()=='nccl'
    trainer=session.trainer;assert trainer.device.type=='cuda' and trainer.step_number>0
    directory=Path(directory)
    def agree(operation):
        try:local=dict(ok=True,value=operation())
        except Exception as error:local=dict(ok=False,error=repr(error))
        results=[None]*dist.get_world_size();dist.all_gather_object(results,local)
        assert all(r['ok'] for r in results),results
        return [r['value'] for r in results]
    # Before any write, reject a reused qualification checkpoint directory on every rank.
    agree(lambda: None if not directory.exists() else (_ for _ in ()).throw(RuntimeError('EXISTING_QUALIFICATION_DIRECTORY')))
    saved_step=trainer.step_number
    torch.cuda.synchronize();start=time.monotonic();session.checkpoint(directory)
    save_seconds=time.monotonic()-start
    # Ensure restoration replaces effective newer state, not an untouched save.
    perturb=QualificationObserver(trainer,session.runtime)
    agree(lambda: (perturb.step(session.step),None)[1])
    torch.cuda.synchronize();start=time.monotonic()
    restored=restore_verified(session,directory/'COMPLETE.json')
    restore_seconds=time.monotonic()-start
    assert trainer.step_number==saved_step
    observer=QualificationObserver(trainer,session.runtime);before=observer.snapshot()
    for _ in range(8):agree(lambda: (observer.step(session.step),None)[1])
    after=observer.snapshot()
    agree(lambda: actual_activation(before,after,8,max_tokens,long_input_receipt))
    local=dict(rank=dist.get_rank(),restored=restored,step=trainer.step_number,
               changed_groups=sorted(observer.changed),save_seconds=save_seconds,
               restore_seconds=restore_seconds,activation=after)
    ranks=[None]*dist.get_world_size();dist.all_gather_object(ranks,local)
    assert all(r['step']==saved_step+8 and r['restored']['step']==saved_step for r in ranks)
    manifest=json.loads((directory/'COMPLETE.json').read_text())
    assert manifest['complete'] and manifest['step']==saved_step
    receipt=dict(complete=True,world_size=dist.get_world_size(),contract_sha256=BASELINE_SHA,
        components=['model','buffers','optimizer','scheduler','ema','rng','data_position','sampler','config','identity'],
        rank_ids=sorted(r['rank'] for r in ranks),effective_continuation_updates=8,
        all_groups_updated=all(set(r['changed_groups'])=={'qwen','bridge','sam_lora','sam_conditions'} for r in ranks),
        checkpoint_manifest_sha256=hashlib.sha256((directory/'COMPLETE.json').read_bytes()).hexdigest(),
        ranks=ranks,qualification_only=True,production_ready=False,full817_validation_completed=False)
    full_restore(receipt,dist.get_world_size())
    return receipt
