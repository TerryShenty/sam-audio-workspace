"""N-rank ablation training entry: one arm, fixed epoch budget, validation after every epoch.

Launched only by scripts/ablation_launch.py (one torchrun inside the user's own Slurm
allocation). Stops at the epoch budget (default 30) or at a checkpoint boundary before
the allocation deadline; continue with --resume <checkpoint> in a new allocation.
"""
import argparse, hashlib, json, os, time
from pathlib import Path


def run_loop(session, output, *, deadline_epoch, epochs, on_step, on_epoch, clock=time.time,
             checkpoint_reserve_seconds=600, step_reserve_seconds=180, checkpoint_every_seconds=1800):
    """V6 production_loop semantics (collective decisions, checkpoint before validation/stop)
    plus an epoch budget: validate after each completed epoch, stop after the last one."""
    from production_loop import collective_decision as decide
    sampler = session.trainer.sampler
    validated = sampler.completed_epochs
    last_saved, last_save_time, overrun = None, clock(), False
    assert deadline_epoch > clock() + checkpoint_reserve_seconds, 'INSUFFICIENT_INITIAL_CHECKPOINT_WINDOW'
    while True:
        now = clock(); remaining = deadline_epoch - now
        finished = validated >= epochs
        proposed = ('stop' if finished or overrun or remaining <= checkpoint_reserve_seconds + step_reserve_seconds
                    else 'validate' if sampler.completed_epochs > validated
                    else 'checkpoint' if now - last_save_time >= checkpoint_every_seconds else 'step')
        action = decide(proposed)
        if action in ('checkpoint', 'stop', 'validate'):
            step = session.trainer.step_number; path = Path(output) / ('step-%012d' % step)
            if last_saved != step:
                started = clock(); session.checkpoint(path)
                assert clock() - started <= checkpoint_reserve_seconds, 'CHECKPOINT_RESERVE_EXCEEDED'
                last_saved, last_save_time = step, clock()
            if action == 'validate':
                validated = sampler.completed_epochs
                on_epoch(validated, path)
                last_save_time = clock()
            if action == 'stop':
                return dict(step=step, checkpoint=str(path), completed_epochs=sampler.completed_epochs,
                            reason='epoch_budget_complete' if validated >= epochs else 'deadline_checkpoint_boundary')
        else:
            started = clock(); result = session.step(); on_step(result)
            if clock() - started > step_reserve_seconds:
                overrun = True


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--plan-sha256', required=True)
    a = p.parse_args()
    raw = a.plan.read_bytes(); assert hashlib.sha256(raw).hexdigest() == a.plan_sha256
    plan = json.loads(raw); v = Path(__file__).resolve().parent; root = Path(plan['root'])
    assert plan['mode'] == 'ablation_train' and str(v) == plan['code_root']
    assert plan['job_id'] == os.environ['SLURM_JOB_ID'] and plan['end_epoch'] == int(os.environ['SLURM_JOB_END_TIME'])
    for name, digest in plan['source_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    from production_bootstrap import install
    install()
    from execution_speed_v29 import install_before_cuda
    install_before_cuda()
    import torch, torch.distributed as dist
    from datetime import timedelta
    rank = int(os.environ['LOCAL_RANK']); world = plan['world_size']
    assert 1 <= world <= 8 and 0 <= rank < world and int(os.environ['WORLD_SIZE']) == world
    assert torch.cuda.device_count() == world, 'Visible GPU count must match requested --gpus'
    torch.cuda.set_device(rank); device = torch.device('cuda', rank)
    assert torch.cuda.get_device_properties(device).total_memory >= 44 * 2**30, 'Requires approximately 48GB per GPU'
    from production_watchdog import WorkerWatchdog, write
    out = Path(plan['output'])
    timeouts = dict(plan['phase_timeouts'])
    watch = WorkerWatchdog(torch, device, rank=rank, output=out, deadline=plan['end_epoch'] - 30, phase_timeouts=timeouts)
    try:
        dist.init_process_group('nccl', timeout=timedelta(seconds=600))
        from ablation_build import build
        session, info = build(root=root, device=device, contract_path=v / 'QWEN_SAM_FAST_BASELINE_V6.json',
                              immutable_identity=plan['deployment_identity'], sam_namespace=plan['sam_namespace'],
                              configuration=plan['effective'], arm=plan['arm'], contrastive_weight=plan['contrastive_weight'])
        for n in plan['runtime_binding']['modules']:
            __import__(n)
        from deployment_guard import source_binding
        source_binding(plan['runtime_binding'], __import__('sys').modules, __file__)
        from watched_session import WatchedSession
        session = WatchedSession(session, watch, phase='train')
        if plan.get('resume'):
            from qualification_restore import restore_verified
            restore_verified(session, Path(plan['resume']))
        for _ in range(3):  # V6: warmup performs real updates and is part of the saved data position
            session.step()
        from portable_codec import load as load_codec
        codec = load_codec(v.parent / 'codec_assets', device)
        rows = info['data']['rows']
        from ablation_validation import evaluate
        from result_csv import training as training_csv
        started = time.monotonic()

        def progress(result):
            loss, norm = result
            training_csv(out / f'training-rank{rank}.csv', rank=rank, step=session.trainer.step_number, loss=float(loss),
                         grad_norm=float(norm), elapsed_seconds=time.monotonic() - started)
            if rank == 0 and session.trainer.step_number % 20 == 0:
                last = session.batch.last or {}
                with (out / 'terms.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=session.trainer.step_number, epoch=session.trainer.sampler.completed_epochs,
                                            terms=last.get('terms').float().cpu().tolist() if last.get('terms') is not None else None,
                                            extra=last.get('extra'))) + '\n')

        def on_epoch(epoch, checkpoint):
            step = session.trainer.step_number
            destination = out / 'validation' / ('epoch-%03d' % epoch)
            summary = evaluate(session, root, rows, destination, step=step, epoch=epoch, codec=codec,
                               batch_size=plan['validation_batch'], watchdog=watch)
            if rank == 0:
                summary.update(checkpoint=str(checkpoint), arm=plan['arm'])
                (destination / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
                with (out / 'epochs.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch, step=step, checkpoint=str(checkpoint),
                                            selection=summary['selection_metric'], headline=summary['headline']), default=str) + '\n')
            dist.barrier()

        result = run_loop(session, out / 'checkpoints', deadline_epoch=plan['end_epoch'] - 1800, epochs=plan['epochs'],
                          on_step=progress, on_epoch=on_epoch)
        if rank == 0:
            write(out / ('TRAIN_COMPLETE.json' if result['reason'] == 'epoch_budget_complete' else 'TRAIN_PAUSED.json'),
                  dict(result, arm=plan['arm']))
    except BaseException as error:
        write(out / f'worker-failure-rank{rank}.json', {'error': repr(error), 'time': time.time()}); raise
    finally:
        watch.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
