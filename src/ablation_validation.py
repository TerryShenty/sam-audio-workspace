"""Per-epoch validation on the two sets, scored and reported SEPARATELY.

  check         200 training videos / 400 A/B targets (seen in training): can the model
                learn the training set? Its floored mean selects the checkpoint.
  gs_benchmark  real General Sound benchmark prompts: reported only.

Generation is the V6 joint_validation.generate path (live weights, fixed per-row noise
seeds, 16-step midpoint ODE, no spans). Scoring is the vendored scorer v2:
no epsilon, no clipping, all-zero prediction = undefined (pred_silent), quiet outputs
scored normally, reference==mixture -> SI-SDRi undefined, plus the -30 dB floored mean.
"""
import json, math, time
from pathlib import Path

SETS = ('check', 'gs_benchmark')
SELECTION = ('check', 'aligned_si_sdr_db_mean_floored')


def _safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_safe(x) for x in v]
    return v


def score_rows(rows, predictions):
    from referring_metrics_v2 import DATASET_TIER, fit_length, load_audio, score_triplet
    out = []
    for row, pred in zip(rows, predictions):
        assert pred['sample_id'] == row['sample_id']
        base = dict(row_id=row['sample_id'], clip_id=row['clip_id'], dataset=row['dataset'], category=row.get('category', ''),
                    gt_tier='true_stem' if row['dataset'] == 'animal3000' else DATASET_TIER.get(row['dataset'], 'unknown'),
                    noise_sha256=pred['noise_sha256'], latent_frames=pred['latent_frames'])
        try:
            base.update(score_triplet(load_audio(row['gt_audio_path']), pred['prediction'], load_audio(row['mixture_audio_path'])))
        except Exception as error:
            base.update(status='error', error=f'{type(error).__name__}: {error}')
        out.append(base)
    return out


def duration(row):
    import soundfile as sf
    info = sf.info(row['mixture_audio_path'])
    return info.frames / info.samplerate


def generate_with_fallback(guarded, part, fallbacks, *, rank, on_attempt, oom, empty_cache):
    """Generate one batch; on OOM split it in half and retry, down to single rows."""
    on_attempt()
    try:
        return guarded([dict(item=r, protocol=r['protocol']) for r in part])
    except oom:
        empty_cache()
        if len(part) == 1:
            raise
        fallbacks.append(dict(rank=rank, size=len(part), first=part[0]['sample_id']))
        half = len(part) // 2
        return (generate_with_fallback(guarded, part[:half], fallbacks, rank=rank, on_attempt=on_attempt, oom=oom, empty_cache=empty_cache)
                + generate_with_fallback(guarded, part[half:], fallbacks, rank=rank, on_attempt=on_attempt, oom=oom, empty_cache=empty_cache))


def evaluate(session, root, rows_by_set, output, *, step, epoch, codec, batch_size=8, watchdog=None):
    """All ranks call this. Returns rank-0 summary dict (None on other ranks).

    Inference batch defaults to 8 per GPU. Each rank's rows run longest first, so memory
    limits show up at the start and padding stays small. On CUDA OOM a batch is split in
    half and retried (down to 1); no row is ever dropped and every fallback is recorded.
    Seeds are per row, so batching never changes which noise a row gets. Note: the V6
    compiled SAM region covers batch 1-2 and 250/256/500 frames; larger or other-length
    batches run the uncompiled SAM path (numerically close, not bitwise identical).
    """
    import torch, torch.distributed as dist
    from joint_validation import evaluation_state, generate
    from referring_metrics_v2 import grouped_summaries
    assert batch_size in (1, 2, 4, 8)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = session.trainer.device; joint = session.runtime.joint
    started = time.time(); local_scores = {}; fallbacks = []; ticks = [0]
    with evaluation_state(joint, device):
        guarded = session.runtime.guard_step(lambda part: generate(joint, codec, session.batch.store, part, device))

        def tick():
            ticks[0] += 1
            if watchdog is not None:
                watchdog.tick('validate', 1_000_000 + ticks[0])

        run = lambda part: generate_with_fallback(guarded, part, fallbacks, rank=rank, on_attempt=tick,
                                                  oom=torch.cuda.OutOfMemoryError, empty_cache=torch.cuda.empty_cache)

        for name in SETS:
            rows = rows_by_set[name]
            mine = sorted((r for i, r in enumerate(rows) if i % world == rank), key=duration, reverse=True)
            scored = []
            for i in range(0, len(mine), batch_size):
                part = mine[i:i + batch_size]
                scored += score_rows(part, run(part))
            local_scores[name] = scored
    gathered = [None] * world
    dist.all_gather_object(gathered, dict(scores=local_scores, fallbacks=fallbacks))
    if rank == 0:
        all_fallbacks = [f for g in gathered for f in g['fallbacks']]
        gathered = [g['scores'] for g in gathered]
    if rank != 0:
        return None
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summary = dict(step=step, epoch=epoch, seconds=time.time() - started, weight_choice='live',
                   solver='midpoint', ode_steps=16, inference_batch=batch_size, oom_batch_splits=all_fallbacks, sets={})
    for name in SETS:
        rows = [r for g in gathered for r in g[name]]
        assert len(rows) == len(rows_by_set[name]) and not any(r.get('status') == 'error' for r in rows), name
        (output / f'{name}_rows.jsonl').write_text(''.join(json.dumps(_safe(r)) + '\n' for r in rows))
        summary['sets'][name] = grouped_summaries(rows, ['overall', 'category', 'gt_tier'])
    overall = {n: next(s for s in summary['sets'][n] if s['group_field'] == 'overall') for n in SETS}
    summary['selection_metric'] = dict(set=SELECTION[0], field=SELECTION[1], value=overall[SELECTION[0]][SELECTION[1]])
    summary['headline'] = {n: {k: overall[n].get(k) for k in (
        'aligned_si_sdr_db_mean_floored', 'aligned_si_sdr_db_floor_rate', 'aligned_si_sdr_db_mean_finite',
        'aligned_si_sdr_db_median', 'si_sdri_db_median', 'failure_rate_sdri_lt_1db', 'silent_rate',
        'si_sdri_undefined', 'rows_scored')} for n in SETS}
    (output / 'summary.json').write_text(json.dumps(_safe(summary), indent=2) + '\n')
    return summary
