"""Sealed four-GPU qualification entry; never a 24-hour production launcher."""
import argparse,hashlib,json,os,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();raw=args.plan.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==args.plan_sha256
    plan=json.loads(raw);root=Path(plan['root']);version=Path(__file__).resolve().parent
    assert str(version)==plan['code_root'] and plan['qualification_only'] is True
    assert plan['world_size']==4 and plan['per_rank_batch']==2 and plan['global_batch']==8
    assert plan['job_id']==os.environ['SLURM_JOB_ID']
    assert plan['end_epoch']==int(os.environ['SLURM_JOB_END_TIME'])
    assert time.time()<plan['end_epoch']-plan['checkpoint_reserve_seconds']
    assert str(Path(__file__).resolve()) in plan['source_files']
    for path,digest in plan['source_files'].items():
        p=Path(path).resolve();assert p.is_relative_to(version)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('SOURCE_CHANGED',path)
    from production_bootstrap import install,resolved_sources
    install()
    from execution_speed_v29 import install_before_cuda
    install_before_cuda()
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    rank=int(os.environ['LOCAL_RANK']);assert 0<=rank<4 and int(os.environ['WORLD_SIZE'])==4
    assert torch.cuda.device_count()==4,'EXACTLY_FOUR_VISIBLE_GPUS_REQUIRED'
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    assert torch.cuda.get_device_properties(device).total_memory>=44*2**30,'Requires approximately 48GB per GPU'
    from production_watchdog import WorkerWatchdog,write
    output=Path(plan['output']);assert output.is_absolute()
    watchdog=WorkerWatchdog(torch,device,rank=rank,output=output,deadline=plan['end_epoch']-180,phase_timeouts=plan['phase_timeouts'])
    try:
        dist.init_process_group('nccl',timeout=timedelta(seconds=plan['collective_timeout_seconds']))
        from production_build import build
        session,evidence=build(root=root,device=device,contract_path=version/'QWEN_SAM_FAST_BASELINE_V6.json',
            immutable_identity=args.plan_sha256,sam_namespace=plan['sam_namespace'],configuration=plan['effective'],
            input_admission_sha256=plan['input_admission_sha256'])
        from deployment_guard import source_binding
        import sys,importlib
        for module in plan['runtime_binding']['modules']:importlib.import_module(module)
        source_binding(plan['runtime_binding'],sys.modules,__file__)
        loaded=resolved_sources()
        for name,record in loaded.items():
            assert plan['source_files'].get(record['path'])==record['sha256'],('UNSEALED_IMPORT',name)
        from qualification_data import attach
        demands=json.loads((root/'TARGET_DEMANDING_CASES_CPU.json').read_text())
        assert hashlib.sha256((root/'TARGET_DEMANDING_CASES_CPU.json').read_bytes()).hexdigest()==plan['demanding_cases_sha256']
        selection=attach(session,demands['cases']['train'])
        from watched_session import WatchedSession
        session=WatchedSession(session,watchdog,phase='qualify')
        from qualification_observer import QualificationObserver
        observer=QualificationObserver(session.trainer,session.runtime)
        before=observer.snapshot();measured=[]
        for case in selection['cases']:
            torch.cuda.reset_peak_memory_stats(device)
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            first.record();observer.step(session.step);last.record();last.synchronize()
            physical=watchdog.nv.nvmlDeviceGetMemoryInfo(watchdog.handle)
            from production_watchdog import cgroup_memory
            host=cgroup_memory()
            measured.append(dict(case=case,sample_ids=session.batch.last['sample_ids'],lengths=session.batch.last['lengths'],
                event_seconds=first.elapsed_time(last)/1000,physical_free=physical.free,physical_used=physical.used,
                allocated_peak=torch.cuda.max_memory_allocated(device),reserved_peak=torch.cuda.max_memory_reserved(device),host=host))
            write(output/f'case-progress-rank{rank}.json',dict(rank=rank,cases=measured,
                observation=observer.snapshot(),runtime=session.runtime.statistics(),
                checkpoint_layers=[i for i,l in enumerate(session.runtime.qwen.qwen.model.layers) if l.gradient_checkpointing],
                model_training=session.runtime.qwen.qwen.model.training))
            assert physical.free>=plan['minimum_physical_free_bytes'] and plan['minimum_physical_free_bytes']>=2*2**30
            assert host['current']<.9*host['limit']
        local_case_record=dict(rank=rank,cases=measured,groups=sorted(observer.changed),before=before,after=observer.snapshot())
        write(output/f'completed-cases-rank{rank}.json',local_case_record)
        ranks=[None]*4;dist.all_gather_object(ranks,local_case_record)
        maximum=max(int(n) for r in ranks for c in r['cases'] for n in c['lengths'])
        assert maximum>=demands['cases']['train']['actual_tokens']['maximum']
        long_input=dict(status='PASS_TARGET_COMPLETE_UPDATES',max_tokens=maximum,retained=23,memo_active=all(r['after']['memo_hits']>r['before']['memo_hits'] for r in ranks),
            physical_headroom_pass=all(c['physical_free']>=plan['minimum_physical_free_bytes'] for r in ranks for c in r['cases']),
            all_ranks_compute=all(r['groups']==['bridge','qwen','sam_conditions','sam_lora'] and all(c['event_seconds']>0 for c in r['cases']) for r in ranks),
            ranks=ranks,physical_memory_measurement='sampled_and_step_boundary_not_continuous_peak')
        write(output/f'long-input-diagnostic-rank{rank}.json',dict(
            status='OBSERVED_BEFORE_ACCEPTANCE_CHECK',long_input=long_input,
            runtime_statistics=session.runtime.statistics(),production_ready=False))
        assert long_input['memo_active'],'ORDINARY_NF4_MEMO_NO_OBSERVED_REPLAY_HITS'
        assert long_input['all_ranks_compute'],'RANK_UPDATE_GROUP_OR_EVENT_EVIDENCE_FAILED'
        assert all(r['after']['retained_layers']==23 and r['after']['retention_fallback_calls']==r['before']['retention_fallback_calls'] for r in ranks)
        assert session.runtime.statistics()['memo']['cap_bytes']==3623878656
        from qualification_run import restore_gate
        receipt=restore_gate(session,output/'restore_checkpoint',max_tokens=maximum,long_input_receipt=long_input)
        from automatic_validation import execute
        execute(session,root,output/'VALIDATION817.json',watchdog,checkpoint_identity=hashlib.sha256((plan['deployment_identity']+':qualification:'+str(session.trainer.step_number)).encode()).hexdigest())
        communication={"status":"DISABLED_IN_PORTABLE_EXPORT"}
        for name,record in resolved_sources().items():
            assert plan['source_files'].get(record['path'])==record['sha256'],('UNSEALED_LATE_IMPORT',name)
        write(output/f'qualification-rank{rank}.json',dict(selection=selection,long_input=long_input,restore=receipt,
            communication=communication,production_ready=False,full817_validation_completed=True))
        dist.barrier()
        if rank==0:write(output/'CHECK_COMPLETE.json',dict(status='QUALIFICATION_FINISHED',production_ready=False,full817_validation_completed=True))
    except BaseException as error:
        write(output/f'worker-failure-rank{rank}.json',dict(error=repr(error),time=time.time(),production_ready=False))
        raise
    finally:
        watchdog.close()
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
