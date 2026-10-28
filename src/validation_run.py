"""Four-rank validation orchestration; real train-state memory approval is required.

The caller supplies the loaded codec and an explicitly qualified batch size.
This does not qualify memory, choose fallback batches, or publish production readiness.
"""
import json,time
from pathlib import Path

def run(session,codec,store,*,root,output,checkpoint_identity,validation_identity,
        expected_step,batch_size,memory_approval,watchdog):
    import torch
    import torch.distributed as dist
    from validation_inputs import bind,batches
    from joint_validation import evaluation_state,generate
    from validation_scoring import load,score_item
    from validation_coverage import collect
    assert dist.is_initialized() and dist.get_world_size()==4 and dist.get_backend()=='nccl'
    started=time.monotonic()
    rank=dist.get_rank();trainer=session.trainer;runtime=session.runtime
    root=Path(root);output=Path(output)
    assert trainer.step_number==expected_step and batch_size in (1,2,4,8)
    # Approval must come from actual padded longest inputs with the full retained
    # training state, codec and scoring active. A config preference is insufficient.
    assert memory_approval['status']=='PASS_TARGET_TRAIN_STATE_VALIDATION_MEMORY'
    assert memory_approval['checkpoint_identity']==checkpoint_identity
    assert memory_approval['validation_identity']==validation_identity
    assert memory_approval['world_size']==4 and memory_approval['batch_size']==batch_size
    assert memory_approval['longest_actual_padded_inputs'] is True
    assert memory_approval['codec_and_scoring_included'] is True
    assert sorted(x['rank'] for x in memory_approval['ranks'])==list(range(4))
    assert all(x['passed'] for x in memory_approval['ranks'])
    protocol=json.loads((root/'VALIDATION_PROTOCOL_BINDING.json').read_text())
    rows=json.loads((root/'BOUND_ROWS.json').read_text())
    items=bind(rows,protocol);plan=batches(items,rank=rank,batch_size=batch_size)
    existing=[None]*4;dist.all_gather_object(existing,output.exists())
    assert not any(existing),'Do not overwrite an existing validation receipt'
    plans=[None]*4;dist.all_gather_object(plans,len(plan));rounds=max(plans)
    namespaces={d:load(root,d) for d in ('music','speech')}
    results=[];joint=runtime.joint;device=trainer.device
    runtime.registry.require_step();runtime.registry.boundary()
    guarded=runtime.guard_step(lambda part:generate(joint,codec,store,part,device))
    with evaluation_state(joint,device):
        for index in range(rounds):
            try:
                if index<len(plan):
                    watchdog.tick('validate',len(results))
                    part=plan[index];predictions=guarded(part)
                    runtime.registry.boundary()
                    assert len(predictions)==len(part)
                    for entry,pred in zip(part,predictions):
                        assert pred['sample_id']==entry['item']['sample_id']
                        score=score_item(namespaces[entry['protocol']['domain']],entry,pred['prediction'])
                        score.update(dataset=entry['item']['dataset'],noise_seed=entry['protocol']['noise_seed'],
                                     noise_sha256=pred['noise_sha256'])
                        assert score['row_index']==entry['protocol']['original_row_index']
                        results.append(score)
                    torch.cuda.synchronize(device)
                    memory=watchdog.nv.nvmlDeviceGetMemoryInfo(watchdog.handle)
                    from production_watchdog import cgroup_memory
                    host=cgroup_memory()
                    assert memory.free>=2*2**30 and host['current']<.9*host['limit'],'VALIDATION_RUNTIME_HEADROOM'
                    watchdog.tick('validate',len(results))
                local=dict(ok=True)
            except Exception as error:local=dict(ok=False,error=repr(error),rank=rank,round=index)
            agreed=[None]*4;dist.all_gather_object(agreed,local)
            assert all(x['ok'] for x in agreed),agreed
    assert trainer.step_number==expected_step
    runtime.registry.require_step();runtime.registry.boundary()
    local=dict(rank=rank,complete=True,rows=results,weight_choice='live',solver='midpoint',steps=16,
               checkpoint_identity=checkpoint_identity,validation_identity=validation_identity)
    ranks=[None]*4;dist.all_gather_object(ranks,local)
    result=collect(ranks,protocol)
    result.update(training_step=expected_step,validation_seconds=time.monotonic()-started)
    published=[None]
    if rank==0:
        try:
            assert not output.exists(),'Do not overwrite an existing validation receipt'
            output.parent.mkdir(parents=True,exist_ok=True)
            temporary=output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
            from result_csv import validation as validation_csv
            validation_csv(output,result)
            published[0]=dict(ok=True)
        except Exception as error:published[0]=dict(ok=False,error=repr(error))
    dist.broadcast_object_list(published,src=0)
    assert published[0]['ok'],published[0]
    return result
