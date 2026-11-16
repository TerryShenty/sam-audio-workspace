"""GPU: build the Qwen processor cache and the frozen Qwen audio/video encoder cache.

Uses the same QwenBridge code paths and cache keys that training reads
(production_data.install_qwen_cache), so no format conversion is involved:
  <data_root>/qwen_cache/processor/..  processor outputs (video at fps 1, max_pixels 100352,
                                       explicit mixture.wav audio, prompt slot)
  <data_root>/qwen_cache/encoders/..   frozen visual / audio tower outputs
Language hidden states are never cached: the language model is stopped before it runs.

Run one process per GPU with --shard i --num-shards N, then once with --finalize.
Existing entries are reused, so an interrupted shard can be rerun.
"""
import argparse, json, os, sys, time
from pathlib import Path
P = Path(__file__).resolve().parent.parent
V = P / 'src'
sys.path.insert(0, str(V))
from ablation_data import key, read_json, write_json


class BeforeLanguage(Exception):
    pass


def paths_for(cache, identity, mk, meta):
    ident = key([identity, mk, meta])
    encoders = {kind: cache / 'encoders' / k[:2] / (k + '.pt') for kind in ('audio', 'video') for k in [key([identity, mk, kind])]}
    return ident, cache / 'processor' / ident[:2] / (ident + '.pt'), encoders


def finalize(root):
    import torch
    media = read_json(root / 'MEDIA.json'); rows = read_json(root / 'ROWS.json')
    receipts = {}
    for f in sorted((root / 'qwen_cache' / 'receipts').glob('shard-*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line); receipts[r['media_key']] = r
    identities = {r['identity'] for r in receipts.values()}
    assert len(identities) == 1, identities
    identity = identities.pop()
    missing = sorted(set(media) - set(receipts))
    assert not missing, f'{len(missing)} media without Qwen cache, e.g. {missing[:3]}'
    records = []
    for mk in sorted(media):
        r = receipts[mk]
        ident, processor_path, encoders = paths_for(root / 'qwen_cache', identity, mk, media[mk])
        assert processor_path.is_file() and all(p.is_file() for p in encoders.values()), mk
        rec = torch.load(processor_path, map_location='cpu', weights_only=True)
        assert rec['identity'] == ident
        records.append(dict(media_key=mk, path=str(processor_path), identity=ident, bytes=processor_path.stat().st_size,
                            base_tokens=r['base_tokens'], max_actual_tokens=r['max_actual_tokens'],
                            video_shape=r['video_shape'], audio_shape=r['audio_shape']))
    write_json(root / 'PROCESSOR_COMPLETE.json', dict(
        status='PASS_CPU_PROCESSOR_ONLY_AV_PENDING', identity=identity, records=records, all_media=True,
        language_hidden_cached=False, max_actual_tokens=max(r['max_actual_tokens'] for r in records)))
    write_json(root / 'QWEN_AV_COMPLETE.json', dict(
        status='PASS_FROZEN_AV_ALL_MEDIA', identity=identity, media_keys=sorted(media), media=len(media),
        components=2 * len(media), all_finite=True, language_hidden_cached=False))
    print('Qwen cache complete:', len(media), 'media; max tokens', max(r['max_actual_tokens'] for r in records))


def main():
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument('--shard', type=int, default=0); a.add_argument('--num-shards', type=int, default=1)
    a.add_argument('--finalize', action='store_true')
    a = a.parse_args()
    paths = read_json(P / 'local_paths.json')
    root = Path(paths['data_root']).resolve(); os.environ['QWEN_DATA_ROOT'] = str(root)
    if a.finalize:
        return finalize(root)
    assert 0 <= a.shard < a.num_shards
    from production_bootstrap import install
    bridge = install()
    import torch
    assert torch.cuda.is_available(), 'Qwen encoder caching runs on one GPU per shard'
    device = torch.device('cuda', 0); torch.cuda.set_device(device)
    media = read_json(root / 'MEDIA.json'); rows = read_json(root / 'ROWS.json')
    qwen = bridge.QwenBridge(device).eval().requires_grad_(False)
    qwen.media = media; qwen.cache = root / 'qwen_cache'
    prompts = {}
    for split in rows.values():
        for r in split:
            n = len(qwen.processor.tokenizer.encode(r['prompt'], add_special_tokens=False))
            prompts[r['media_key']] = max(prompts.get(r['media_key'], 0), n)
    calls = {'n': 0}

    def stop(*args, **kwargs):
        calls['n'] += 1
        raise BeforeLanguage()

    handle = qwen.qwen.model.register_forward_pre_hook(stop, with_kwargs=True)
    out = root / 'qwen_cache' / 'receipts' / f'shard-{a.shard:03d}-of-{a.num_shards:03d}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['media_key'] for line in out.read_text().splitlines()} if out.exists() else set()
    mine = [mk for mk in sorted(media) if int(key(mk), 16) % a.num_shards == a.shard and mk not in done]
    started = time.time()
    try:
        for i, mk in enumerate(mine):
            before = calls['n']
            with torch.inference_mode():
                try:
                    qwen(mk, 'the target sound')
                except BeforeLanguage:
                    pass
                else:
                    raise RuntimeError('Language interception failed')
            assert calls['n'] == before + 1
            qwen._av_memory.clear()
            ident, processor_path, encoders = paths_for(qwen.cache, qwen.identity, mk, media[mk])
            assert processor_path.is_file() and all(p.is_file() for p in encoders.values())
            rec = qwen.prepared(mk)
            base = rec['x']['input_ids'].shape[-1]
            row = dict(media_key=mk, identity=qwen.identity, base_tokens=base,
                       max_actual_tokens=base - rec['slot_length'] + prompts[mk],
                       video_shape=list(rec['x']['pixel_values_videos'].shape), audio_shape=list(rec['x']['input_features'].shape))
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
            if (i + 1) % 25 == 0 or i + 1 == len(mine):
                print(json.dumps(dict(shard=a.shard, done=i + 1, total=len(mine), seconds=round(time.time() - started, 1))), flush=True)
    finally:
        handle.remove()
    print('Shard finished. After every shard: python scripts/build_qwen_cache.py --finalize')


if __name__ == '__main__':
    main()
