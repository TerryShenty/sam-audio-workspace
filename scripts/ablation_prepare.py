"""CPU only: build the ablation data contract from the downloaded raw data.

Inputs
  --animal     snapshot of the public animal-3000 dataset (queries.jsonl + media/)
  --benchmark  snapshot of the (private) referring benchmark at the pinned revision
  --out        data root (MEDIA.json, ROWS.json, SAM_CACHE_PLAN.json, base_identity.json)

Validation = two separately reported sets:
  check         200 training videos (400 A/B targets) that ARE also trained on; it checks
                that the model can learn the training set. Checkpoint selection uses this set.
  gs_benchmark  every General Sound prompt of the benchmark (real videos, reported only).
No GPU, no model weights, no network.
"""
import argparse, csv, json, random, sys
from pathlib import Path
P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'src'))
from ablation_data import (CHECK_SEED, CHECK_VIDEOS, clean_key, key, mixture_key, read_json,
                           sha256_file, video_key, write_json)

GS_NAMES = {'general_sound', 'generalsound', 'gs', 'general sound'}
EXPECTED_GS = dict(prompts=413, targets=323, videos=181)  # benchmark revision 54c3fda5


def extract_audio(mp4, wav):
    """Decode a video's audio track to a mono float32 48 kHz WAV (the GS mixture)."""
    import av, numpy as np, soundfile as sf
    chunks = []
    with av.open(str(mp4)) as container:
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=48000)
        for frame in container.decode(audio=0):
            chunks.extend(o.to_ndarray() for o in resampler.resample(frame))
        chunks.extend(o.to_ndarray() for o in resampler.resample(None))
    assert chunks, f'no audio in {mp4}'
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.concatenate(chunks, axis=1)[0], 48000, subtype='FLOAT')


def main():
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument('--animal', type=Path, required=True)
    a.add_argument('--benchmark', type=Path, required=True)
    a.add_argument('--benchmark-revision', default='54c3fda5')
    a.add_argument('--out', type=Path, required=True)
    a = a.parse_args()
    animal, bench, out = a.animal.resolve(), a.benchmark.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    media, plan, rows = {}, {}, {'train': [], 'check': [], 'gs_benchmark': []}
    digest = {}

    def sha(path):
        path = str(Path(path).resolve())
        if path not in digest:
            digest[path] = sha256_file(path)
        return digest[path]

    def add_component(mix_path, video_path, target_path, residual):
        mk = mixture_key(sha(mix_path)); vk = video_key(sha(video_path), sha(mix_path))
        res_id = sha(residual) if residual else ['mixture_minus_target', sha(mix_path)]
        ck = clean_key(sha(target_path), res_id)
        spec = plan.setdefault(mk, {'path': str(Path(mix_path).resolve()), 'videos': {}, 'targets': {}})
        spec['videos'][vk] = str(Path(video_path).resolve())
        spec['targets'][ck] = {'target': str(Path(target_path).resolve()),
                               'residual': str(Path(residual).resolve()) if residual else None}
        return {'mixture': mk, 'video': vk, 'clean': ck}

    # ---- animal 3000: 6,000 A/B queries, mixture = target_A + target_B
    queries = [json.loads(line) for line in (animal / 'queries.jsonl').read_text().splitlines() if line.strip()]
    assert len(queries) == 6000, len(queries)
    by_video = {}
    for q in queries:
        by_video.setdefault(q['video_id'], {})[q['role']] = q
    assert len(by_video) == 3000 and all(set(v) == {'A', 'B'} for v in by_video.values())
    for vid in sorted(by_video):
        pair = by_video[vid]
        folder = animal / 'media' / vid
        video, mix = folder / 'video.mp4', folder / 'mixture.wav'
        mkey = 'animal3000::' + vid
        media[mkey] = {'video': {'path': str(video), 'sha256': sha(video)}, 'audio': {'path': str(mix), 'sha256': sha(mix)}}
        for role, other in (('A', 'B'), ('B', 'A')):
            q = pair[role]
            refs = add_component(mix, video, folder / f'target_{role}.wav', folder / f'target_{other}.wav')
            rows['train'].append(dict(sample_id=f'animal3000::{vid}::{role}', split='animal_train', media_key=mkey,
                                      prompt=q['prompt'], refs=refs, pair_id=vid, role=role))
    # ---- check set: 200 training videos, both targets, fixed noise seeds
    chosen = sorted(random.Random(CHECK_SEED).sample(sorted(by_video), CHECK_VIDEOS))
    train_by_id = {r['sample_id']: r for r in rows['train']}
    for vid in chosen:
        for role in ('A', 'B'):
            r = dict(train_by_id[f'animal3000::{vid}::{role}'])
            folder = animal / 'media' / vid
            r.update(sample_id=r['sample_id'] + '::check', split='animal_check_validation', dataset='animal3000',
                     clip_id=vid, category=by_video[vid][role].get('mode', ''),
                     gt_audio_path=str(folder / f'target_{role}.wav'), mixture_audio_path=str(folder / 'mixture.wav'),
                     protocol={'noise_seed': 20261110 + 1009 * len(rows['check'])})
            rows['check'].append(r)
    # ---- benchmark General Sound (real videos)
    with (bench / 'metadata' / 'prompts.csv').open(newline='', encoding='utf-8-sig') as f:
        table = [r for r in csv.DictReader(f) if r.get('dataset', '').strip().lower() in GS_NAMES]
    counts = dict(prompts=len(table), targets=len({(r['clip_id'], r['target_source_id']) for r in table}),
                  videos=len({r['clip_id'] for r in table}))
    if a.benchmark_revision.startswith('54c3fda5'):
        assert counts == EXPECTED_GS, ('Unexpected GS benchmark size', counts)
    for i, r in enumerate(table):
        video = bench / r['video_path']; gt = bench / r['gt_audio_path']
        mix = out / 'gs_mixture' / (r['clip_id'] + '.wav')
        if not mix.exists():
            extract_audio(video, mix)
        mkey = 'gs::' + r['clip_id']
        media.setdefault(mkey, {'video': {'path': str(video), 'sha256': sha(video)}, 'audio': {'path': str(mix), 'sha256': sha(mix)}})
        refs = add_component(mix, video, gt, None)
        rows['gs_benchmark'].append(dict(
            sample_id=f"gs::{r['clip_id']}::{r['target_source_id']}::{i}", split='gs_benchmark_validation',
            media_key=mkey, prompt=r['prompt'], refs=refs, dataset='general_sound', clip_id=r['clip_id'],
            category=r.get('category') or r.get('task_type', ''), gt_audio_path=str(gt), mixture_audio_path=str(mix),
            protocol={'noise_seed': 20261111 + 1009 * i}))
    models = read_json(P / 'configs' / 'MODEL_SHA256.json')
    write_json(out / 'base_identity.json', {'qwen': models['qwen'], 'note': 'model-file identity from configs/MODEL_SHA256.json'})
    write_json(out / 'MEDIA.json', media)
    write_json(out / 'ROWS.json', rows)
    write_json(out / 'SAM_CACHE_PLAN.json', {'mixtures': plan})
    summary = dict(status='PASS_CPU_DATA_CONTRACT', train_rows=len(rows['train']), check_rows=len(rows['check']),
                   check_videos=CHECK_VIDEOS, gs_rows=len(rows['gs_benchmark']), gs_counts=counts,
                   media=len(media), mixtures=len(plan),
                   cleans=sum(len(v['targets']) for v in plan.values()),
                   benchmark_revision=a.benchmark_revision, identity=key([read_json(out / 'ROWS.json'), read_json(out / 'MEDIA.json')]))
    write_json(out / 'DATA_CONTRACT.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
