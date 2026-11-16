"""Data contract for the animal-3000 / GS ablation (replaces the fixed 111K V6 contract).

Data root (written by scripts/ablation_prepare.py, then filled by the two cache builders):
  MEDIA.json             media_key -> {video: {path, sha256}, audio: {path, sha256}}
                         (exactly the record QwenBridge.prepared() hashes and keys on)
  ROWS.json              {train: [...], check: [...], gs_benchmark: [...]}
  SAM_CACHE_PLAN.json    mixtures/videos/cleans to encode into the SAM namespace
  base_identity.json     Qwen model-file identity (from configs/MODEL_SHA256.json)
  SAM_CACHE_COMPLETE.json, PROCESSOR_COMPLETE.json, QWEN_AV_COMPLETE.json   (cache builders)

Row fields: sample_id, split, media_key, prompt, refs{mixture,video,clean}; training rows add
pair_id and role (A/B); validation rows add protocol{noise_seed}, gt_audio_path,
mixture_audio_path, dataset, clip_id, category.
"""
import contextlib, hashlib, json
from pathlib import Path

SPLITS = ('train', 'check', 'gs_benchmark')
CHECK_VIDEOS = 200
CHECK_SEED = 20261110


def key(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mixture_key(audio_sha):
    return key(['sam-mixture-v1', audio_sha])


def video_key(video_sha, audio_sha):
    # PE frames are sampled at the mixture latent rate, so the mixture identity is part of the key.
    return key(['sam-video-v1', video_sha, audio_sha])


def clean_key(target_sha, residual):
    return key(['sam-clean-v1', target_sha, residual])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def load(root):
    """Load and cross-check the prepared contract; fails closed on any missing cache stage."""
    root = Path(root)
    names = ['MEDIA.json', 'ROWS.json', 'SAM_CACHE_PLAN.json', 'base_identity.json',
             'SAM_CACHE_COMPLETE.json', 'PROCESSOR_COMPLETE.json', 'QWEN_AV_COMPLETE.json']
    for n in names:
        assert (root / n).is_file(), f'Missing {n}: run prepare and both cache builders first'
    media = read_json(root / 'MEDIA.json'); rows = read_json(root / 'ROWS.json')
    processor = read_json(root / 'PROCESSOR_COMPLETE.json'); av = read_json(root / 'QWEN_AV_COMPLETE.json')
    sam = read_json(root / 'SAM_CACHE_COMPLETE.json')
    assert set(rows) == set(SPLITS)
    assert sam['status'] == 'PASS_SAM_CACHE_ALL_COMPONENTS'
    assert processor['status'] == 'PASS_CPU_PROCESSOR_ONLY_AV_PENDING' and processor['all_media']
    assert av['status'] == 'PASS_FROZEN_AV_ALL_MEDIA' and av['all_finite']
    assert processor['identity'] == av['identity']
    assert {r['media_key'] for r in processor['records']} == set(media) == set(av['media_keys'])
    train = rows['train']
    assert len({r['sample_id'] for split in SPLITS for r in rows[split]}) == sum(len(rows[s]) for s in SPLITS)
    for r in train:
        assert r['split'] == 'animal_train' and r['role'] in ('A', 'B') and r['media_key'] in media
    for s in ('check', 'gs_benchmark'):
        for r in rows[s]:
            assert r['split'].endswith('_validation') and r['media_key'] in media and 'noise_seed' in r['protocol']
    hashes = {n: sha256_file(root / n) for n in names}
    identity = key(hashes)
    return dict(media=media, rows=rows, train=train, processor=processor, hashes=hashes, identity=identity)


class AblationBatch:
    """Bound A/B rows -> AblationJoint call (V6 JointBatch semantics, 2 rows per rank)."""

    def __init__(self, rows, store, device):
        self.items = list(rows); self.store = store; self.device = device; self.last = None

    def __call__(self, wrapped, indices):
        import torch
        from production_data import tree
        assert len(indices) == 2
        items = [self.items[i] for i in indices]
        assert items[0]['pair_id'] == items[1]['pair_id'] and [x['role'] for x in items] == ['A', 'B']
        samples = [self.store.assemble(item) for item in items]
        args = [tree(a, lambda t: t.to(self.device, non_blocking=True)) for a, _ in samples]
        clean = [target.to(self.device, non_blocking=True) for _, target in samples]
        for a, target in zip(args, clean):
            assert target.shape[-1] == 256 and 'text_features' not in a and 'text_mask' not in a
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else contextlib.nullcontext()
        with context:
            loss, terms, lengths = wrapped(items, args, clean, [.5, .5])
        extra = getattr(wrapped.module, 'last_terms', None)
        self.last = dict(sample_ids=[r['sample_id'] for r in items], lengths=list(lengths), terms=terms.detach(),
                         extra={k: v.detach().float().cpu().tolist() for k, v in (extra or {}).items()})
        return loss
