"""GPU: encode the SAM-side cache (DAC mixture latent, PE video features, target+residual latent).

Layout matches the V6 ProductionStore: <sam_namespace>/{mixture,video,clean}/<key[:2]>/<key>.pt.
Run one process per GPU with --shard i --num-shards N (shards are disjoint by mixture),
then once with --finalize on CPU. Existing files are reused, so an interrupted shard can
simply be rerun. Requires prepare (local_paths.json) and scripts/ablation_prepare.py.
"""
import argparse, json, os, sys, time
from pathlib import Path
P = Path(__file__).resolve().parent.parent
V = P / 'src'
sys.path[:0] = [str(V), str(V / 'upstream_v2/runtime_v1/code')]
from ablation_data import read_json, write_json


def path(root, kind, k):
    return Path(root) / kind / k[:2] / (k + '.pt')


def put(root, kind, k, value):
    import torch
    p = path(root, kind, k); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp'); torch.save(value, tmp); tmp.replace(p)


def finalize(data_root, namespace):
    plan = read_json(data_root / 'SAM_CACHE_PLAN.json')['mixtures']
    missing = []
    for mk, spec in plan.items():
        missing += [('mixture', mk)] if not path(namespace, 'mixture', mk).is_file() else []
        missing += [('video', k) for k in spec['videos'] if not path(namespace, 'video', k).is_file()]
        missing += [('clean', k) for k in spec['targets'] if not path(namespace, 'clean', k).is_file()]
    assert not missing, f'{len(missing)} SAM components missing, e.g. {missing[:3]}'
    write_json(data_root / 'SAM_CACHE_COMPLETE.json', dict(
        status='PASS_SAM_CACHE_ALL_COMPONENTS', namespace=str(namespace), mixtures=len(plan),
        videos=sum(len(s['videos']) for s in plan.values()), cleans=sum(len(s['targets']) for s in plan.values())))
    print('SAM cache complete:', len(plan), 'mixtures')


def main():
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument('--shard', type=int, default=0); a.add_argument('--num-shards', type=int, default=1)
    a.add_argument('--finalize', action='store_true')
    a = a.parse_args()
    paths = read_json(P / 'local_paths.json')
    data_root = Path(paths['data_root']); namespace = Path(paths['sam_namespace'])
    if a.finalize:
        return finalize(data_root, namespace)
    assert 0 <= a.shard < a.num_shards
    import torch
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.native_finetuning.feature_cache import FrozenExtractor
    from sam_audio.native_finetuning.optimized_utils import StageTimer
    from sam_audio.native_finetuning.data import _fit_to_length
    assert torch.cuda.is_available(), 'SAM cache encoding runs on one GPU per shard'
    device = torch.device('cuda', 0)
    cfg = read_json(P / 'configs' / 'sam_tuning.json')
    sam = SAMAudio.from_pretrained(paths['sam_model'], visual_ranker=None, text_ranker=None, span_predictor=None, strict=True)
    processor = SAMAudioProcessor.from_pretrained(paths['sam_model'])
    sam.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    extractor = FrozenExtractor(sam, processor, device, torch.bfloat16, StageTimer(device, enabled=False),
                                to_cpu=True, video_frame_chunk=int(cfg.get('video_frame_chunk', 64)))
    plan = read_json(data_root / 'SAM_CACHE_PLAN.json')['mixtures']
    mine = [k for k in sorted(plan) if int(k, 16) % a.num_shards == a.shard]
    started = time.time()
    for i, mk in enumerate(mine):
        spec = plan[mk]
        wave = None
        if not path(namespace, 'mixture', mk).is_file() or any(not path(namespace, 'clean', c).is_file() for c in spec['targets']):
            wave = extractor.read_audio(spec['path'])
        if not path(namespace, 'mixture', mk).is_file():
            put(namespace, 'mixture', mk, extractor.mixture(wave))
        mixture = torch.load(path(namespace, 'mixture', mk), map_location='cpu', weights_only=True)
        assert mixture['forward_args']['anchor_ids'].tolist() == [[0, 3]], 'Unexpected default anchors'
        for vk, video in spec['videos'].items():
            if not path(namespace, 'video', vk).is_file():
                put(namespace, 'video', vk, extractor.video(video, mixture))
        for ck, t in spec['targets'].items():
            if path(namespace, 'clean', ck).is_file():
                continue
            n = mixture['num_samples']
            target = _fit_to_length(extractor.read_audio(t['target']), n)
            residual = _fit_to_length(extractor.read_audio(t['residual']), n) if t['residual'] else wave[..., :n] - target
            clean = extractor.clean(target, residual)
            assert clean.shape[-1] == 256 and clean.shape[:2] == mixture['forward_args']['audio_features'].shape[:2]
            put(namespace, 'clean', ck, clean)
        if (i + 1) % 50 == 0 or i + 1 == len(mine):
            print(json.dumps(dict(shard=a.shard, done=i + 1, total=len(mine), seconds=round(time.time() - started, 1))), flush=True)
    print('Shard finished. After every shard: python scripts/build_sam_cache.py --finalize')


if __name__ == '__main__':
    main()
