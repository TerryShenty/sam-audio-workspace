"""Uncapped N-rank training on prepared data; direct 24-hour entry with in-run safeguards."""
import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args()
    raw=a.plan.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.plan_sha256
    plan=json.loads(raw);v=Path(__file__).resolve().parent;root=Path(plan['root'])
    assert plan['mode']=='train' and not plan['qualification_only'] and str(v)==plan['code_root']
    assert plan['job_id']==os.environ['SLURM_JOB_ID'] and plan['end_epoch']==int(os.environ['SLURM_JOB_END_TIME'])
    assert time.time()<plan['end_epoch']-900
    for name,digest in plan['source_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    from production_bootstrap import install
    install()
    from execution_speed_v29 import install_before_cuda
    install_before_cuda()
    import torch,torch.distributed as dist
    from datetime import timedelta
    rank=int(os.environ['LOCAL_RANK']);world=plan['world_size']
    assert 1<=world<=8 and 0<=rank<world and int(os.environ['WORLD_SIZE'])==world
    assert torch.cuda.device_count()==world,'Visible GPU count must match requested --gpus'
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    assert torch.cuda.get_device_properties(device).total_memory>=44*2**30,'Requires approximately 48GB per GPU'
    from production_watchdog import WorkerWatchdog,write
    out=Path(plan['output']);watch=WorkerWatchdog(torch,device,rank=rank,output=out,deadline=plan['end_epoch']-30,phase_timeouts=plan['phase_timeouts'])
    try:
        dist.init_process_group('nccl',timeout=timedelta(seconds=600))
        from production_build import build
        session,_=build(root=root,device=device,contract_path=v/'QWEN_SAM_FAST_BASELINE_V6.json',immutable_identity=plan['deployment_identity'],sam_namespace=plan['sam_namespace'],configuration=plan['effective'],input_admission_sha256=plan['input_admission_sha256'])
        for n in plan['runtime_binding']['modules']:importlib.import_module(n)
        from deployment_guard import source_binding
        source_binding(plan['runtime_binding'],sys.modules,__file__)
        from watched_session import WatchedSession
        session=WatchedSession(session,watch,phase='train')
        if plan.get('resume'):
            from qualification_restore import restore_verified
            restore_verified(session,Path(plan['resume']))
        from half_epoch import HalfEpoch
        schedule=HalfEpoch(len(session.trainer.sampler.ids),session.trainer.sampler.cursor)
        # Warmup performs real production updates and is included in the saved data position.
        for _ in range(3):session.step()
        from production_loop import run,collective_decision
        from result_csv import training as training_csv
        training_started=time.monotonic()
        def progress(result):
            loss,norm=result
            training_csv(out/f"training-rank{rank}.csv",rank=rank,step=session.trainer.step_number,loss=float(loss),grad_norm=float(norm),elapsed_seconds=time.monotonic()-training_started)
            write(out/f'progress-rank{rank}.json',{'step':session.trainer.step_number,'loss':float(loss),'grad_norm':float(norm),'time':time.time(),'full817_validation_completed':False})
        from automatic_validation import execute
        def periodic(checkpoint):
            step=session.trainer.step_number
            destination=out/'validation'/('step-%012d.json'%step)
            identity=hashlib.sha256((checkpoint/'COMPLETE.json').read_bytes()).hexdigest()
            if destination.exists():
                saved=json.loads(destination.read_text())
                assert saved['status']=='PASS_COMPLETE_817_SCORE_COVERAGE' and saved['checkpoint_identity']==identity and saved['training_step']==step
            else:
                execute(session,root,destination,watch,checkpoint_identity=identity)
            schedule.complete(session.trainer.sampler.cursor)
        result=run(session,out/'checkpoints' ,deadline_epoch=plan['end_epoch']-1860,checkpoint_reserve_seconds=600,step_reserve_seconds=120,checkpoint_every_seconds=900,decide=collective_decision,on_step=progress,validation_due=lambda:schedule.due(session.trainer.sampler.cursor),on_validation=periodic)
        # Checkpoint is complete before validation; failure preserves it.
        from automatic_validation import execute
        final_identity=hashlib.sha256((Path(result['checkpoint'])/'COMPLETE.json').read_bytes()).hexdigest()
        previous=out/'validation'/('step-%012d.json'%session.trainer.step_number)
        if previous.exists():
            saved=json.loads(previous.read_text())
            assert saved['checkpoint_identity']==final_identity and saved['status']=='PASS_COMPLETE_817_SCORE_COVERAGE'
            if rank==0:
                write(out/'VALIDATION817.json',saved)
                from result_csv import validation as validation_csv
                validation_csv(out/'VALIDATION817.json',saved)
            dist.barrier()
        else:execute(session,root,out/'VALIDATION817.json',watch,checkpoint_identity=final_identity)
        if rank==0:write(out/'TRAIN_COMPLETE.json',dict(result,full817_validation_completed=True,validation=str(out/'VALIDATION817.json')))
    except BaseException as error:
        write(out/f'worker-failure-rank{rank}.json',{'error':repr(error),'time':time.time()});raise
    finally:
        watch.close()
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
