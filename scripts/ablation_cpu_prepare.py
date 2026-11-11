"""CPU-only preparation for the ablation (replaces scripts/prepare.py, which is bound to the 111K data).

Checks model files against configs/MODEL_SHA256.json, prepares the frozen codec assets,
builds the V6 CPU extensions, verifies the sealed V6 runtime, and records local paths.
Never submits a job or downloads anything.
"""
import argparse, hashlib, importlib, json, os, sys
from pathlib import Path
P = Path(__file__).resolve().parent.parent; V = P / 'src'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2, sort_keys=True) + '\n')


def main():
    sys.path.insert(0, str(P / 'scripts'))
    from verify_package import verify
    verify()
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument('--data-root', type=Path, required=True, help='output of scripts/ablation_prepare.py')
    a.add_argument('--model-root', type=Path, required=True, help='contains Qwen3-Omni-30B-A3B-Thinking and sam-audio-large-tv')
    a = a.parse_args()
    assert not os.environ.get('SLURM_JOB_GPUS'), 'Run this on CPU resources'
    data = a.data_root.expanduser().resolve(); models = a.model_root.expanduser().resolve()
    contract = json.loads((data / 'DATA_CONTRACT.json').read_text())
    assert contract['status'] == 'PASS_CPU_DATA_CONTRACT'
    paths = {'data_root': str(data), 'sam_namespace': str(data / 'sam_namespace'),
             'qwen_model': str(models / 'Qwen3-Omni-30B-A3B-Thinking'), 'sam_model': str(models / 'sam-audio-large-tv')}
    for kind, records in json.loads((P / 'configs/MODEL_SHA256.json').read_text()).items():
        for name, digest in records.items():
            assert sha(Path(paths[kind + '_model']) / name) == digest, ('Model mismatch', kind, name)
    write(P / 'local_paths.json', paths); os.environ['QWEN_DATA_ROOT'] = str(data)
    sys.path.insert(0, str(V)); from production_bootstrap import install
    install()
    from dependency_seal import verify as verify_seal
    verify_seal(V)
    import torch
    from portable_codec import prepare as prepare_codec
    prepare_codec(paths['sam_model'], P / 'codec_assets')
    assert not torch.cuda.is_initialized()
    natives = []
    for name in ['unique_scatter_round3', 'bulk_gather_support_round6', 'pointwise_moe_round8', 'buffers_moe_round9']:
        m = importlib.import_module(name); e = m.extension(); p = Path(e.__file__).resolve()
        assert p.is_relative_to(V)
        natives.append({'module': name, 'path': str(p), 'sha256': sha(p)})
    assert not torch.cuda.is_initialized()
    write(V / 'native_extensions.json', natives)
    from deployment_guard import configuration, GROUPS
    configuration(V / 'QWEN_SAM_FAST_BASELINE_V6.json', json.loads((P / 'configs/effective_config.json').read_text()), GROUPS)
    import importlib.metadata as metadata
    env = {n: metadata.version(n) for n in ('torch', 'transformers', 'bitsandbytes', 'triton', 'numpy')}
    write(P / 'CPU_READY.json', {'status': 'ABLATION_CPU_PREPARED_TARGET_GPU_UNTESTED', 'paths_sha256': sha(P / 'local_paths.json'),
                                 'data_contract_sha256': sha(data / 'DATA_CONTRACT.json'), 'native_sha256': sha(V / 'native_extensions.json'),
                                 'environment': env, 'codec_admission_sha256': sha(P / 'codec_assets/CODEC_READY.json')})
    print('CPU preparation complete. Next: build the SAM and Qwen caches on GPU (see README).')


if __name__ == '__main__':
    main()
