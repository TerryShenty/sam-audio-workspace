"""Referring-separation scorer v2 (2026-11-10), vendored with the -30 dB floored mean.

Copied from the research repository tools/referring_metrics_v2.py; the only
addition is the floored-mean summary (FLOOR_DB) used for checkpoint selection.

Replaces the scoring *definition* used by ``tools/hf_jobs_score_benchmark_all8.py``
(protocol ``sam_audio_referring_all8_si_activity_rts_v1_20260903``) without
editing that file, which stays as the historical record.

Defects fixed relative to v1 (see reports/SAM Audio 提升方向评估.md):

1. v1 computed ``10*log10((num+EPS)/(den+EPS))``.  An all-zero prediction then
   scores exactly 0 dB, which is *higher* than most real outputs, and was
   averaged into the mean.  Shrinking an unrelated waveform toward zero also
   drifts its score toward 0 dB.  v2 uses no epsilon in the ratio.
   Quiet predictions (v1 rule: < -70 dBFS or < -30 dB vs GT) are now scored
   normally like any other output (user decision 2026-11-10: models do not emit
   true silence); the v1 ``silent_failure`` flag is kept only as a diagnostic
   column.  Only a numerically all-zero prediction, for which SI-SDR is
   undefined, gets status ``pred_silent``; it counts as a failure and ranks
   below every scored prediction in medians.
2. v1 clipped scores to [-100, 100].  v2 does not clip; non-finite outcomes are
   given explicit statuses instead.
3. v1 averaged everything.  v2 reports median/IQR, mean over finite scored rows,
   failure rates at two thresholds (all-zero = failure), all-zero rate, quiet-flag rate and a
   clip-level cluster-bootstrap CI for the median SI-SDRi.
4. A reference that is identical to the mixture (near-GT mask covering the
   whole clip) gives an infinite mixture SI-SDR, so SI-SDRi is undefined.  v2
   marks it ``sdri_undefined_reference_equals_mixture`` instead of producing
   -inf or a clipped number.
5. Every row carries a GT tier (true_stem / close_mic / near_gt_enhanced /
   near_gt_mask) and optional ``gt_source`` so that headline numbers can exclude
   near-GT-mask and SAM-cleaned references.  Which subsets enter the headline is
   a protocol decision recorded in the output, not hard-coded here.

The dataset revision is a required input and is written into the protocol.  This
module never alters a prediction using ground truth.

Usage (pure CPU)::

    python tools/referring_metrics_v2.py score --manifest rows.csv \
        --dataset-revision 54c3fda5... --out-dir OUT
    python tools/referring_metrics_v2.py reaggregate --table per_prompt.csv \
        --group-field model --out-dir OUT

``rows.csv`` needs: row_id, clip_id, dataset, prediction_path, reference_path,
mixture_path (WAV, or MP4 if PyAV is installed); optional: system, task_type,
category, gt_source, seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

PROTOCOL_ID = "sam_audio_referring_metrics_v2_20261110"
TARGET_SR = 48_000
MAX_ALIGNMENT_MS = 50.0
# Silence thresholds are kept identical to v1 so silent_failure flags match.
SILENT_ABSOLUTE_DBFS = -70.0
SILENT_RELATIVE_TO_GT_DB = -30.0
# Energy below this (per-sample mean square) is treated as numerically zero.
ZERO_ENERGY = 1e-20
FAILURE_THRESHOLDS_DB = (0.0, 1.0)
# Floored mean (user decision 2026-11-10): aligned SI-SDR below FLOOR_DB counts
# as FLOOR_DB, including all-zero predictions; this is the checkpoint-selection
# number for the GS/animal ablation. Per-row values themselves are never clipped.
FLOOR_DB = -30.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261110

DATASET_TIER = {
    "urmp": "true_stem",
    "choralsynth": "true_stem",
    "easycom": "close_mic",
    "misp": "close_mic",
    "msdwild": "near_gt_enhanced",
    "general_sound": "near_gt_mask",
    "generalsound": "near_gt_mask",
}

STATUS_OK = "ok"
STATUS_PRED_SILENT = "pred_silent"
STATUS_REF_SILENT = "ref_silent_excluded"
STATUS_ZERO_PROJECTION = "zero_projection"
STATUS_PERFECT = "perfect_reconstruction"


# ---------------------------------------------------------------- primitives

def mean_square(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x)) if x.size else 0.0


def dbfs(x: np.ndarray) -> float:
    ms = mean_square(x)
    return 10.0 * math.log10(ms) if ms > 0 else -math.inf


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(audio) >= length:
        return audio[:length].copy()
    return np.pad(audio, (0, length - len(audio)))


def is_silent(prediction: np.ndarray, reference: np.ndarray) -> bool:
    """v1-compatible silent-prediction rule (absolute or relative to GT)."""
    p, r = dbfs(prediction), dbfs(reference)
    if p == -math.inf:
        return True
    return p < SILENT_ABSOLUTE_DBFS or (p - r) < SILENT_RELATIVE_TO_GT_DB


def align_small_lag(reference: np.ndarray, estimate: np.ndarray,
                    max_ms: float = MAX_ALIGNMENT_MS) -> tuple[np.ndarray, int]:
    """Same search as v1: max |xcorr| within +/-max_ms, zero-padded shift."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if mean_square(reference) <= ZERO_ENERGY or mean_square(estimate) <= ZERO_ENERGY:
        return estimate.copy(), 0
    corr = correlate(estimate - estimate.mean(), reference - reference.mean(), mode="full", method="fft")
    lags = correlation_lags(len(estimate), len(reference), mode="full")
    keep = np.abs(lags) <= int(round(TARGET_SR * max_ms / 1000.0))
    lag = int(lags[keep][int(np.argmax(np.abs(corr[keep])))])
    out = np.zeros_like(estimate)
    if lag > 0:
        out[:-lag] = estimate[lag:]
    elif lag < 0:
        out[-lag:] = estimate[:lag]
    else:
        out[:] = estimate
    return out, lag


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> tuple[float | None, str]:
    """Zero-mean SI-SDR without epsilon or clipping.

    Returns (value_db, status).  value is None unless status is ok/perfect
    (perfect -> +inf) or zero_projection (-> -inf).  Silence is decided by the
    caller with :func:`is_silent` because it is relative to the reference.
    """
    s = np.asarray(reference, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    s = s - s.mean()
    e = e - e.mean()
    ref_energy = float(np.dot(s, s))
    if ref_energy / max(len(s), 1) <= ZERO_ENERGY:
        return None, STATUS_REF_SILENT
    if float(np.dot(e, e)) / max(len(e), 1) <= ZERO_ENERGY:
        return None, STATUS_PRED_SILENT
    alpha = float(np.dot(e, s)) / ref_energy
    target = alpha * s
    noise = e - target
    num = float(np.dot(target, target))
    den = float(np.dot(noise, noise))
    if num == 0.0:
        return -math.inf, STATUS_ZERO_PROJECTION
    if den == 0.0:
        return math.inf, STATUS_PERFECT
    return 10.0 * math.log10(num / den), STATUS_OK


def sd_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Scale-dependent SDR (Le Roux et al. 2019); penalises gain errors."""
    s = np.asarray(reference, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    num = float(np.dot(s, s))
    den = float(np.dot(s - e, s - e))
    if num == 0.0:
        return None
    if den == 0.0:
        return math.inf
    return 10.0 * math.log10(num / den)


# ------------------------------------------------------------ per-row score

def score_triplet(reference: np.ndarray, prediction: np.ndarray, mixture: np.ndarray) -> dict[str, Any]:
    """Score one prediction.  All arrays mono at TARGET_SR; trimmed/padded to GT."""
    n = len(reference)
    prediction = fit_length(prediction, n)
    mixture = fit_length(mixture, n)
    row: dict[str, Any] = {
        "prediction_dbfs": dbfs(prediction),
        "prediction_vs_gt_db": dbfs(prediction) - dbfs(reference),
    }
    ref_value, ref_status = si_sdr(reference, reference)
    if ref_status == STATUS_REF_SILENT:
        row.update(status=STATUS_REF_SILENT, silent_failure=False)
        return row
    silent = is_silent(prediction, reference)
    aligned, lag = align_small_lag(reference, prediction)
    mix_aligned, mix_lag = align_small_lag(reference, mixture)
    mix_value, mix_status = si_sdr(reference, mix_aligned)
    row.update(alignment_lag_ms=lag * 1000.0 / TARGET_SR,
               mixture_alignment_lag_ms=mix_lag * 1000.0 / TARGET_SR,
               mixture_aligned_si_sdr_db=mix_value, mixture_status=mix_status,
               silent_failure=bool(silent))
    # Quiet-but-nonzero predictions are scored normally; silent_failure is a
    # diagnostic flag only.  si_sdr() itself returns pred_silent for all-zero.
    value, status = si_sdr(reference, aligned)
    if status == STATUS_PRED_SILENT:
        row.update(status=STATUS_PRED_SILENT, aligned_si_sdr_db=None, si_sdri_db=None,
                   aligned_sd_sdr_db=None)
        return row
    row.update(status=status, aligned_si_sdr_db=value, aligned_sd_sdr_db=sd_sdr(reference, aligned))
    if mix_status == STATUS_PERFECT:
        row["si_sdri_db"] = None
        row["si_sdri_status"] = "sdri_undefined_reference_equals_mixture"
    elif value is None or mix_value is None:
        row["si_sdri_db"] = None
        row["si_sdri_status"] = f"sdri_undefined_{mix_status}"
    else:
        row["si_sdri_db"] = value - mix_value
        row["si_sdri_status"] = "ok"
    return row


# ---------------------------------------------------------------- aggregation

def _rank_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    """Values for rank statistics: silent / zero-projection -> -inf."""
    out = []
    for r in rows:
        status = r.get("status")
        if status in (STATUS_PRED_SILENT, STATUS_ZERO_PROJECTION):
            out.append(-math.inf)
            continue
        v = _num(r.get(field))
        if v is not None:
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _median_ci(values_by_cluster: dict[str, list[float]], seed: int = BOOTSTRAP_SEED) -> tuple[float | None, float | None]:
    clusters = [np.asarray(v, dtype=np.float64) for v in values_by_cluster.values() if v]
    if len(clusters) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, len(clusters), len(clusters))
        medians[i] = np.median(np.concatenate([clusters[j] for j in pick]))
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return _finite_or_none(lo), _finite_or_none(hi)


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def summarize(rows: list[dict[str, Any]], with_ci: bool = True) -> dict[str, Any]:
    total = len(rows)
    excluded = [r for r in rows if r.get("status") == STATUS_REF_SILENT]
    scored = [r for r in rows if r.get("status") != STATUS_REF_SILENT]
    silent = sum(r.get("status") == STATUS_PRED_SILENT for r in scored)
    out: dict[str, Any] = {
        "rows": total, "rows_scored": len(scored), "rows_ref_silent_excluded": len(excluded),
        "silent_rate": silent / len(scored) if scored else None,
        "quiet_flag_rate": (sum(str(r.get("silent_failure", "")).lower() == "true" for r in scored) / len(scored)) if scored else None,
    }
    for field in ("aligned_si_sdr_db", "si_sdri_db"):
        ranked = _rank_values(scored, field)
        finite = ranked[np.isfinite(ranked)]
        out[f"{field}_n"] = int(ranked.size)
        if ranked.size:
            q25, q50, q75 = np.percentile(ranked, [25, 50, 75]) if np.isfinite(ranked).all() else _inf_safe_quantiles(ranked)
            out[f"{field}_median"] = _finite_or_none(q50)
            out[f"{field}_p25"] = _finite_or_none(q25)
            out[f"{field}_p75"] = _finite_or_none(q75)
        out[f"{field}_mean_finite"] = float(finite.mean()) if finite.size else None
        out[f"{field}_n_finite"] = int(finite.size)
    floored, floor_hits, perfect = [], 0, 0
    for r in scored:
        status = r.get("status")
        v = -math.inf if status in (STATUS_PRED_SILENT, STATUS_ZERO_PROJECTION) else _num(r.get("aligned_si_sdr_db"))
        if v is None:
            continue
        if v == math.inf:
            perfect += 1
            continue
        if v < FLOOR_DB:
            floor_hits += 1
        floored.append(max(v, FLOOR_DB))
    out["aligned_si_sdr_db_floor"] = FLOOR_DB
    out["aligned_si_sdr_db_mean_floored"] = float(np.mean(floored)) if floored else None
    out["aligned_si_sdr_db_floor_rate"] = floor_hits / len(floored) if floored else None
    out["aligned_si_sdr_db_perfect_excluded_from_floored_mean"] = perfect
    sdri = _rank_values(scored, "si_sdri_db")
    undefined = sum(1 for r in scored if r.get("status") == STATUS_OK and _num(r.get("si_sdri_db")) is None)
    out["si_sdri_undefined"] = undefined
    if sdri.size:
        out["si_sdri_positive_fraction"] = float(np.mean(sdri > 0))
        for t in FAILURE_THRESHOLDS_DB:
            out[f"failure_rate_sdri_lt_{t:g}db"] = float(np.mean(sdri < t))
    if with_ci and sdri.size:
        clusters: dict[str, list[float]] = defaultdict(list)
        for r in scored:
            v = -math.inf if r.get("status") in (STATUS_PRED_SILENT, STATUS_ZERO_PROJECTION) else _num(r.get("si_sdri_db"))
            if v is not None:
                clusters[str(r.get("clip_id", r.get("row_id")))].append(v)
        lo, hi = _median_ci(clusters)
        out["si_sdri_median_ci95_lo"], out["si_sdri_median_ci95_hi"] = lo, hi
    return out


def _inf_safe_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    ordered = np.sort(values)
    def q(p: float) -> float:
        # nearest-rank on a sorted array keeps -inf meaningful
        return float(ordered[min(len(ordered) - 1, max(0, int(math.ceil(p * len(ordered))) - 1))])
    return q(0.25), q(0.5), q(0.75)


def grouped_summaries(rows: list[dict[str, Any]], fields: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for field in fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(field, "")) if field != "overall" else "overall"].append(r)
        for key, members in sorted(groups.items()):
            out.append({"group_field": field, "group": key, **summarize(members)})
    return out


# ------------------------------------------------------------------- I/O

def load_audio(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}:
        return _decode_video_audio(p)
    import soundfile as sf
    audio, sr = sf.read(str(p), always_2d=True, dtype="float64")
    audio = audio.mean(axis=1)
    if int(sr) != TARGET_SR:
        g = math.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g)
    return np.asarray(audio, dtype=np.float64)


def _decode_video_audio(path: Path) -> np.ndarray:
    import av  # optional dependency, only needed for MP4 mixtures
    chunks = []
    with av.open(str(path)) as container:
        resampler = None
        for frame in container.decode(audio=0):
            if resampler is None:
                resampler = av.AudioResampler(format="fltp", layout=frame.layout.name, rate=TARGET_SR)
            chunks.extend(o.to_ndarray() for o in resampler.resample(frame))
        if resampler is not None:
            chunks.extend(o.to_ndarray() for o in resampler.resample(None))
    if not chunks:
        raise RuntimeError(f"no audio decoded from {path}")
    return np.concatenate(chunks, axis=1).mean(axis=0).astype(np.float64)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def protocol_record(**extra: Any) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": TARGET_SR,
        "alignment": f"max |xcorr| within +/-{MAX_ALIGNMENT_MS:g} ms (same as v1)",
        "si_sdr": "zero-mean SI-SDR, no epsilon in ratio, no clipping",
        "silence": f"quiet predictions (< {SILENT_ABSOLUTE_DBFS} dBFS or < {SILENT_RELATIVE_TO_GT_DB} dB vs GT, v1 rule) are scored normally and only flagged (silent_failure, quiet_flag_rate); a numerically all-zero prediction has undefined SI-SDR: status pred_silent, counted as failure, ranked -inf, excluded from finite means",
        "reference_silent": "row excluded and counted",
        "si_sdri_undefined": "when reference equals mixture (mixture SI-SDR infinite)",
        "failure_thresholds_db": list(FAILURE_THRESHOLDS_DB),
        "floored_mean": f"aligned SI-SDR mean with values < {FLOOR_DB:g} dB (and all-zero predictions) counted as {FLOOR_DB:g}; floor_rate = share of rows below the floor; per-row values unclipped",
        "median_ci": f"clip-level cluster bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
        "gt_tiers": DATASET_TIER,
        "headline_subset": "not fixed by code; report per tier and an explicit headline selection",
        **extra,
    }


def cmd_score(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    rows_in = _read_csv(manifest_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}

    def get(path: str) -> np.ndarray:
        if path not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[path] = load_audio(path)
        return cache[path]

    rows_out = []
    for i, r in enumerate(rows_in, 1):
        base = {k: r.get(k, "") for k in ("row_id", "system", "seed", "clip_id", "dataset", "task_type", "category", "gt_source")}
        base["gt_tier"] = r.get("gt_tier") or DATASET_TIER.get(r.get("dataset", "").lower(), "unknown")
        try:
            pred = get(r["prediction_path"])
            base["prediction_sha256"] = hashlib.sha256(Path(r["prediction_path"]).read_bytes()).hexdigest()
            base.update(score_triplet(get(r["reference_path"]), pred, get(r["mixture_path"])))
        except Exception as exc:  # recorded, never silently dropped
            base.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows_out.append(base)
        if i % 200 == 0:
            print(f"[{i}/{len(rows_in)}]", flush=True)
    _finish(rows_out, out_dir, args, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())


def cmd_reaggregate(args: argparse.Namespace) -> None:
    """Re-summarise an existing v1 per-row table with v2 statistics.

    Waveforms are not re-read, so v1 SI-SDR values are reused.  Rows whose v1
    aligned SI-SDR is exactly 0.0 *and* flagged silent (all-zero signature) or
    at the +/-100 clip are flagged.  Quiet rows with a real value keep it.
    """
    table = Path(args.table)
    rows = []
    for r in _read_csv(table):
        r = dict(r)
        v = _num(r.get("aligned_si_sdr_db"))
        silent_flag = str(r.get("silent_failure", "")).lower() == "true"
        r["legacy_flag"] = ""
        if silent_flag and v == 0.0:
            # v1 eps/eps signature of an all-zero prediction: undefined in v2.
            r["status"] = STATUS_PRED_SILENT
            r["legacy_flag"] = "v1_exact_zero_silent"
        elif v is not None and abs(v) >= 100.0:
            r["status"] = STATUS_ZERO_PROJECTION if v < 0 else STATUS_PERFECT
            r["legacy_flag"] = "v1_clipped"
        else:
            r["status"] = STATUS_OK
        r.setdefault("gt_tier", DATASET_TIER.get(r.get("dataset", "").lower(), "unknown"))
        rows.append(r)
    _finish(rows, Path(args.out_dir), args, source_table=str(table),
            source_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
            note="reaggregation of v1 per-row values; SI-SDR not recomputed from audio")


def _finish(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace, **extra: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["overall", *[f for f in (args.group_field or []) if f]]
    summaries = grouped_summaries([r for r in rows if r.get("status") != "error"], fields)
    _write_csv(out_dir / "rows_v2.csv", rows)
    _write_csv(out_dir / "summary_v2.csv", summaries)
    proto = protocol_record(dataset_revision=getattr(args, "dataset_revision", None),
                            row_count=len(rows), error_rows=sum(r.get("status") == "error" for r in rows), **extra)
    (out_dir / "protocol_v2.json").write_text(json.dumps(_json_safe(proto), indent=2, ensure_ascii=False))
    print(json.dumps(_json_safe(proto), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset-revision", required=True,
                   help="benchmark commit the references come from (e.g. 54c3fda5...)")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--group-field", action="append", default=["system", "dataset", "gt_tier", "category"])
    s.set_defaults(func=cmd_score)
    a = sub.add_parser("reaggregate")
    a.add_argument("--table", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--group-field", action="append", default=[])
    a.set_defaults(func=cmd_reaggregate)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
