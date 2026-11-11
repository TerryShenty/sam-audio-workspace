"""One torchrun, N persistent ranks for ONE ablation arm inside your own Slurm allocation."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
P = Path(__file__).resolve().parent.parent; V = P / 'src'
ARMS = ('baseline', 'dual_t0', 'contrastive', 'dual_t0_contrastive')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(P / 'scripts'))
    from verify_package import verify
    verify()
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--arm', choices=ARMS, required=True)
    a.add_argument('--gpus', type=int, default=4)
    a.add_argument('--epochs', type=int, default=30)
    a.add_argument('--contrastive-weight', type=float, default=1.0)
    a.add_argument('--validation-batch', type=int, default=1, choices=(1, 2, 4, 8))
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--resume', type=Path, help='checkpoint directory from an earlier run of the SAME arm and GPU count')
    a = a.parse_args()
    assert 1 <= a.gpus <= 8, 'GPUs must be 1-8 on one node'
    assert a.epochs >= 1
    assert os.environ.get('SLURM_JOB_ID'), 'Must run inside your own Slurm allocation'
    end = int(os.environ['SLURM_JOB_END_TIME']); assert end - time.time() > 3600, 'Need at least one hour of allocation'
    paths = json.loads((P / 'local_paths.json').read_text()); ready = json.loads((P / 'CPU_READY.json').read_text())
    assert ready['status'] == 'ABLATION_CPU_PREPARED_TARGET_GPU_UNTESTED'
    assert ready['paths_sha256'] == sha(P / 'local_paths.json')
    assert ready['native_sha256'] == sha(V / 'native_extensions.json')
    assert ready['codec_admission_sha256'] == sha(P / 'codec_assets/CODEC_READY.json')
    root = Path(paths['data_root']); os.environ['QWEN_DATA_ROOT'] = str(root)
    assert ready['data_contract_sha256'] == sha(root / 'DATA_CONTRACT.json')
    assert not a.output.exists(), 'Use a new output directory, including for resume'
    a.output.mkdir(parents=True); out = a.output.resolve()
    files = {str(p.resolve()): sha(p) for p in V.rglob('*.py')}
    identity = hashlib.sha256((sha(P / 'SOURCE_MANIFEST.json') + ready['paths_sha256'] + ready['data_contract_sha256']
                               + ready['native_sha256'] + ready['codec_admission_sha256']).encode()).hexdigest()
    effective = json.loads((P / 'configs/effective_config.json').read_text())
    effective.update(world_size=a.gpus, per_rank_batch=2, global_batch=2 * a.gpus)
    plan = {'root': str(root), 'code_root': str(V), 'output': str(out), 'job_id': os.environ['SLURM_JOB_ID'], 'end_epoch': end,
            'mode': 'ablation_train', 'world_size': a.gpus, 'per_rank_batch': 2, 'global_batch': 2 * a.gpus,
            'effective': effective, 'source_files': files, 'deployment_identity': identity, 'sam_namespace': paths['sam_namespace'],
            'arm': a.arm, 'contrastive_weight': a.contrastive_weight if 'contrastive' in a.arm else 0.0,
            'epochs': a.epochs, 'validation_batch': a.validation_batch,
            'phase_timeouts': {'construct': 900, 'qualify': 900, 'train': 180, 'checkpoint': 600, 'restore': 600, 'validate': 600}}
    if a.resume:
        plan['resume'] = str(a.resume.resolve())
    sys.path.insert(0, str(V)); from production_bootstrap import PATHS
    names = ['bp_backend', 'bp_dispatch', 'bp_live', 'bp_registry', 'bp_lifecycle', 'bytepair_launch', 'bytepair_kernel', 'word8_launch',
             'word8_decode_kernel', 'word4_launch', 'word4_decode_kernel', 'nf4_attention_policy', 'nf4_attention_backward_policy',
             'a30_adapter', 'int8_fused_rowquant_linear', 'int8_fused_dynamic_kernel', 'int8_rowquant_kernel']
    modules = {}
    for n in names:
        p = next(r / (n + '.py') for r in PATHS if (r / (n + '.py')).is_file()); modules[n] = {'path': str(p), 'sha256': sha(p)}
    from deployment_guard import BASELINE_SHA
    entry = V / 'ablation_train.py'
    plan['runtime_binding'] = {'modules': modules, 'entry': str(entry), 'entry_sha256': sha(entry), 'contract_sha256': BASELINE_SHA}
    planpath = out / 'plan.json'; planpath.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Arm {a.arm}: {a.gpus} GPUs, per-GPU batch 2 (one A/B mixture), global batch {2 * a.gpus}, {a.epochs} epochs')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=' + str(a.gpus),
           str(entry), '--plan', str(planpath), '--plan-sha256', sha(planpath)]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
