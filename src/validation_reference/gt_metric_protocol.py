# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "av>=14,<19",
#   "huggingface-hub>=0.36,<2",
#   "numpy>=2,<3",
#   "requests>=2.32,<3",
#   "scipy>=1.14,<2",
#   "soundfile>=0.13,<1",
# ]
# ///
"""Score all eight saved SAM Audio candidates against benchmark references.

This is an evaluation-only Hugging Face Job.  It never runs SAM Audio and never
uses ground truth to alter a generated waveform.  For every prompt/candidate it
computes the repository's audited Activity F1 and SI-SDR protocol, plus SI-SDRi
against the original mixture.  Official residual target suppression (RTS) is
computed only when the model-produced residual WAV exists; it is deliberately
not approximated as ``mixture - target``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import av
import numpy as np
import requests
import soundfile as sf
from huggingface_hub import CommitOperationAdd, HfApi
from scipy.signal import correlate, correlation_lags, resample_poly


DATASET_ID = "sam-audio-research/sam-audio-referring-benchmark"
DATASET_REVISION = "8a6d4a19b4a277019d4c0950fcb8fc16d4f838b8"
DATASET_ROOT = Path("/dataset")
EXPECTED_MANIFEST_SHA256 = "f31e11abf195f1112c242295d0b2130f43c75aa754bd72db968d0a6a25c1012a"
RESULT_REPO = "TerryShenty/sam-audio-jobs-results"
DEFAULT_SOURCE_RUN = "sam-audio-benchmark-full-20260902"
DEFAULT_METRIC_RUN = "all8_gt_metrics_v1_20260903"
SOURCE_SHARD = "shard_000_of_001"

EXPECTED_PROMPTS = 2225
EXPECTED_CANDIDATES = 8
TARGET_SR = 48_000
MAX_ALIGNMENT_MS = 50.0
ACTIVITY_FRAME_MS = 100.0
ACTIVITY_HOP_MS = 50.0
ACTIVITY_RELATIVE_TO_GT_P95_DB = -35.0
ACTIVITY_ABSOLUTE_FLOOR_DBFS = -60.0
NEAR_GT_COLLAR_MS = 250.0
NEAR_GT_MAX_AMBIGUOUS_BURST_MS = 300.0
SILENT_RELATIVE_TO_GT_DB = -30.0
SILENT_ABSOLUTE_DBFS = -70.0
RTS_REPORT_FLOOR_DB = -60.0
RTS_REPORT_CEILING_DB = 60.0
TRUE_GT_DATASETS = {"urmp", "choralsynth"}
NEAR_GT_DATASETS = {"msdwild", "easycom", "general_sound"}
EPS = 1e-12

CANDIDATE_FIELDS = [
    "status", "row_index", "candidate_number", "dataset", "domain",
    "task_type", "clip_id", "video_path", "gt_audio_path",
    "target_source_id", "prompt", "reference_tier", "visual_winner",
    "text_winner", "fusion_winner", "prediction_sha256", "prediction_rms_dbfs",
    "prediction_vs_gt_rms_db", "silent_failure", "alignment_lag_samples",
    "alignment_lag_ms", "alignment_corr", "si_sdr_db", "aligned_si_sdr_db",
    "mixture_alignment_lag_samples", "mixture_alignment_lag_ms",
    "mixture_alignment_corr", "mixture_aligned_si_sdr_db", "si_sdri_db",
    "activity_primary_metric", "activity_primary_f1",
    "activity_precision_strict", "activity_recall_strict", "activity_f1_strict",
    "activity_precision_tolerant", "activity_recall_tolerant",
    "activity_f1_tolerant", "activity_ignored_ambiguous_frames",
    "activity_ignored_ambiguous_ms", "activity_threshold_dbfs",
    "activity_prediction_level_match_db", "rts_status", "rts_db", "rts_raw_db",
    "rts_target_remaining_ratio", "rts_target_remaining_percent",
    "rts_active_frames", "error_type", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--metric-run-id", default=DEFAULT_METRIC_RUN)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0


def db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) >= length:
        return audio[:length].copy()
    return np.pad(audio, (0, length - len(audio)))


def resample_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        # soundfile is samples x channels; PyAV is normalized before this call.
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if int(sample_rate) != TARGET_SR:
        divisor = math.gcd(int(sample_rate), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // divisor, int(sample_rate) // divisor)
    return np.asarray(audio, dtype=np.float64)


def wav_bytes_to_mono(payload: bytes) -> np.ndarray:
    audio, sample_rate = sf.read(io.BytesIO(payload), always_2d=True, dtype="float64")
    return resample_mono(audio, int(sample_rate))


def decode_mixture_mp4(payload: bytes) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        handle.write(payload)
        handle.flush()
        with av.open(handle.name) as container:
            if not container.streams.audio:
                raise RuntimeError("Video has no audio stream")
            resampler = None
            for frame in container.decode(audio=0):
                if resampler is None:
                    resampler = av.AudioResampler(
                        format="fltp", layout=frame.layout.name, rate=TARGET_SR
                    )
                for output in resampler.resample(frame):
                    chunks.append(output.to_ndarray())
            if resampler is not None:
                for output in resampler.resample(None):
                    chunks.append(output.to_ndarray())
    if not chunks:
        raise RuntimeError("Video audio decoded to zero samples")
    # PyAV arrays are channels x samples.  Preserve the benchmark convention of
    # evaluating mono audio by averaging channels.
    audio = np.concatenate(chunks, axis=1).mean(axis=0)
    return np.asarray(audio, dtype=np.float64)


def shift_estimate_to_reference(estimate: np.ndarray, lag_samples: int) -> np.ndarray:
    estimate = np.asarray(estimate, dtype=np.float64)
    output = np.zeros_like(estimate)
    lag_samples = int(lag_samples)
    if lag_samples > 0:
        output[:-lag_samples] = estimate[lag_samples:]
    elif lag_samples < 0:
        amount = -lag_samples
        output[amount:] = estimate[:-amount]
    else:
        output[:] = estimate
    return output


def align_small_lag(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, int, float]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if rms(reference) < EPS or rms(estimate) < EPS:
        return estimate.copy(), 0, 0.0
    ref_zm = reference - reference.mean()
    est_zm = estimate - estimate.mean()
    corr = correlate(est_zm, ref_zm, mode="full", method="fft")
    lags = correlation_lags(len(est_zm), len(ref_zm), mode="full")
    max_lag = int(round(TARGET_SR * MAX_ALIGNMENT_MS / 1000.0))
    keep = np.abs(lags) <= max_lag
    local_corr = corr[keep]
    local_lags = lags[keep]
    winner = int(np.argmax(np.abs(local_corr)))
    lag = int(local_lags[winner])
    denom = np.linalg.norm(ref_zm) * np.linalg.norm(est_zm)
    coefficient = float(local_corr[winner] / max(float(denom), EPS))
    return shift_estimate_to_reference(estimate, lag), lag, coefficient


def standard_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    estimate = np.asarray(estimate, dtype=np.float64) - np.mean(estimate)
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= EPS:
        raise RuntimeError("Cannot compute SI-SDR against a silent GT")
    scale = float(np.dot(estimate, reference)) / reference_energy
    projected = scale * reference
    distortion = estimate - projected
    numerator = float(np.dot(projected, projected))
    denominator = float(np.dot(distortion, distortion))
    score = 10.0 * math.log10((numerator + EPS) / (denominator + EPS))
    return float(np.clip(score, -100.0, 100.0))


def frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    starts = np.arange(0, len(audio) - frame_length + 1, hop_length)
    values = np.asarray([rms(audio[s : s + frame_length]) for s in starts])
    return starts, values


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.size:
        return mask.copy()
    return np.convolve(mask.astype(np.int32), np.ones(2 * radius + 1), mode="same") > 0


def binary_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def prf(reference_active: np.ndarray, prediction_active: np.ndarray) -> tuple[float, float, float, int, int, int]:
    tp = int(np.sum(reference_active & prediction_active))
    fp = int(np.sum(~reference_active & prediction_active))
    fn = int(np.sum(reference_active & ~prediction_active))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def activity_metrics(reference: np.ndarray, aligned_estimate: np.ndarray, reference_tier: str, silent_failure: bool) -> dict[str, Any]:
    frame_len = int(round(TARGET_SR * ACTIVITY_FRAME_MS / 1000.0))
    hop_len = int(round(TARGET_SR * ACTIVITY_HOP_MS / 1000.0))
    _, ref_env = frame_rms(reference, frame_len, hop_len)
    _, est_env_raw = frame_rms(aligned_estimate, frame_len, hop_len)
    positive_ref = ref_env[ref_env > EPS]
    positive_est = est_env_raw[est_env_raw > EPS]
    if not positive_ref.size:
        raise RuntimeError("GT has no measurable activity")
    ref_p95 = float(np.percentile(positive_ref, 95))
    threshold = max(
        10.0 ** (ACTIVITY_ABSOLUTE_FLOOR_DBFS / 20.0),
        ref_p95 * 10.0 ** (ACTIVITY_RELATIVE_TO_GT_P95_DB / 20.0),
    )
    est_p95 = float(np.percentile(positive_est, 95)) if positive_est.size else EPS
    est_scale = ref_p95 / max(est_p95, EPS)
    ref_active = ref_env >= threshold
    pred_active = est_env_raw * est_scale >= threshold
    if silent_failure:
        pred_active[:] = False
    strict = prf(ref_active, pred_active)
    collar_frames = int(math.ceil(NEAR_GT_COLLAR_MS / ACTIVITY_HOP_MS))
    gt_tolerant = dilate_binary(ref_active, collar_frames)
    pred_tolerant = dilate_binary(pred_active, collar_frames)
    ignored = np.zeros_like(pred_active, dtype=bool)
    ignored_ms = 0.0
    if reference_tier == "near_gt":
        for start, end in binary_runs(pred_active):
            run_frames = end - start
            duration_ms = ACTIVITY_FRAME_MS + max(0, run_frames - 1) * ACTIVITY_HOP_MS
            if duration_ms <= NEAR_GT_MAX_AMBIGUOUS_BURST_MS and not gt_tolerant[start:end].any():
                ignored[start:end] = True
                ignored_ms += duration_ms
    scored_prediction = pred_active & ~ignored
    p_den = int(scored_prediction.sum())
    r_den = int(ref_active.sum())
    tolerant_precision = int((scored_prediction & gt_tolerant).sum()) / p_den if p_den else 0.0
    tolerant_recall = int((ref_active & pred_tolerant).sum()) / r_den if r_den else 0.0
    tolerant_f1 = 2 * tolerant_precision * tolerant_recall / (tolerant_precision + tolerant_recall) if tolerant_precision + tolerant_recall else 0.0
    if reference_tier == "near_gt":
        primary_name, primary_f1 = "collar_tolerant_activity_f1", tolerant_f1
    else:
        primary_name, primary_f1 = "strict_activity_f1", strict[2]
    return {
        "activity_primary_metric": primary_name,
        "activity_primary_f1": float(primary_f1),
        "activity_precision_strict": float(strict[0]),
        "activity_recall_strict": float(strict[1]),
        "activity_f1_strict": float(strict[2]),
        "activity_precision_tolerant": float(tolerant_precision),
        "activity_recall_tolerant": float(tolerant_recall),
        "activity_f1_tolerant": float(tolerant_f1),
        "activity_ignored_ambiguous_frames": int(ignored.sum()),
        "activity_ignored_ambiguous_ms": float(ignored_ms),
        "activity_threshold_dbfs": float(db(threshold)),
        "activity_prediction_level_match_db": float(db(est_scale)),
    }


def target_projected_energy(reference: np.ndarray, signal: np.ndarray) -> tuple[float, int]:
    frame_len = int(round(TARGET_SR * ACTIVITY_FRAME_MS / 1000.0))
    hop_len = int(round(TARGET_SR * ACTIVITY_HOP_MS / 1000.0))
    if len(reference) < frame_len:
        reference = np.pad(reference, (0, frame_len - len(reference)))
        signal = np.pad(signal, (0, frame_len - len(signal)))
    starts, ref_env = frame_rms(reference, frame_len, hop_len)
    positive = ref_env[ref_env > EPS]
    if not positive.size:
        raise RuntimeError("GT has no measurable active frame for RTS")
    threshold = max(
        10.0 ** (ACTIVITY_ABSOLUTE_FLOOR_DBFS / 20.0),
        float(np.percentile(positive, 95)) * 10.0 ** (ACTIVITY_RELATIVE_TO_GT_P95_DB / 20.0),
    )
    energy, used = 0.0, 0
    for start in starts[ref_env >= threshold]:
        target = reference[start : start + frame_len]
        source = signal[start : start + frame_len]
        target = target - target.mean()
        source = source - source.mean()
        target_energy = float(np.dot(target, target))
        if target_energy <= EPS:
            continue
        inner = float(np.dot(source, target))
        energy += inner * inner / target_energy
        used += 1
    if not used:
        raise RuntimeError("No valid active frame for RTS")
    return float(energy), used


def rts_from_energies(mixture_energy: float, residual_energy: float) -> tuple[float, float, float]:
    if mixture_energy <= EPS or residual_energy < 0:
        raise RuntimeError("Invalid target-projected energy for RTS")
    raw = 10.0 * math.log10((mixture_energy + EPS) / (residual_energy + EPS))
    return float(np.clip(raw, RTS_REPORT_FLOOR_DB, RTS_REPORT_CEILING_DB)), float(raw), float(residual_energy / mixture_energy)


def self_tests() -> None:
    rng = np.random.default_rng(20260724)
    target = rng.normal(size=TARGET_SR)
    mixture = target + rng.normal(size=TARGET_SR)
    base, _ = target_projected_energy(target, mixture)
    half, _ = target_projected_energy(target, 0.5 * mixture)
    tenth, _ = target_projected_energy(target, 0.1 * mixture)
    if not np.isclose(rts_from_energies(base, half)[0], 6.020599913, atol=1e-6):
        raise AssertionError("RTS half-amplitude self-test failed")
    if not np.isclose(rts_from_energies(base, tenth)[0], 20.0, atol=1e-6):
        raise AssertionError("RTS tenth-amplitude self-test failed")
    if not np.isclose(standard_si_sdr(target, mixture) - standard_si_sdr(target, mixture), 0.0, atol=1e-10):
        raise AssertionError("SI-SDRi self-test failed")


class HubReader:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def get(self, repo_id: str, repo_type: str, revision: str, path: str, optional: bool = False) -> bytes | None:
        namespace = "datasets/" if repo_type == "dataset" else ""
        encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
        url = f"https://huggingface.co/{namespace}{repo_id}/resolve/{quote(revision, safe='')}/{encoded_path}"
        response = self.session.get(url, timeout=(30, 300))
        if optional and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content


def source_item_dir(source_run: str, row_index: int) -> str:
    return f"benchmark_runs/{source_run}/{SOURCE_SHARD}/items/prompt_{row_index:06d}"


def read_csv_bytes(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_local_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        output = float(value)
        return output if math.isfinite(output) else None
    except (TypeError, ValueError):
        return None


def selection_row(method: str, prompt: dict[str, Any], candidates: list[dict[str, Any]], candidate_number: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_index": prompt["row_index"], "dataset": prompt["dataset"],
        "task_type": prompt["task_type"], "method": method,
        "candidate_number": candidate_number or "", "status": "unavailable",
    }
    if candidate_number is None:
        return row
    match = next((x for x in candidates if int(x["candidate_number"]) == candidate_number and x["status"] == "success"), None)
    if match is None:
        return row
    row.update(
        status="success",
        activity_primary_f1=match["activity_primary_f1"],
        activity_f1_strict=match["activity_f1_strict"],
        aligned_si_sdr_db=match["aligned_si_sdr_db"],
        si_sdri_db=match["si_sdri_db"],
        rts_status=match["rts_status"], rts_db=match["rts_db"],
        silent_failure=match["silent_failure"],
    )
    return row


def summarize_values(values: Iterable[Any]) -> dict[str, Any]:
    numbers = np.asarray([x for value in values if (x := safe_float(value)) is not None], dtype=np.float64)
    if not numbers.size:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(numbers.size), "mean": float(numbers.mean()),
        "median": float(np.median(numbers)), "p25": float(np.percentile(numbers, 25)),
        "p75": float(np.percentile(numbers, 75)),
    }


def build_aggregate_rows(selection_rows: list[dict[str, Any]], group_field: str | None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selection_rows:
        if row.get("status") != "success":
            continue
        group = str(row.get(group_field, "overall")) if group_field else "overall"
        groups[(group, str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (group, method), rows in sorted(groups.items()):
        record: dict[str, Any] = {"group": group, "method": method, "prompt_count": len(rows)}
        for metric in ["activity_primary_f1", "activity_f1_strict", "aligned_si_sdr_db", "si_sdri_db", "rts_db"]:
            stats = summarize_values(row.get(metric) for row in rows)
            for name, value in stats.items():
                record[f"{metric}_{name}"] = value
        sdri = [x for row in rows if (x := safe_float(row.get("si_sdri_db"))) is not None]
        record["si_sdri_positive_fraction"] = float(np.mean(np.asarray(sdri) > 0)) if sdri else None
        record["silent_failure_fraction"] = float(np.mean([str(row.get("silent_failure")).lower() == "true" for row in rows]))
        output.append(record)
    return output


def upload_checkpoint(api: HfApi, output_root: Path, remote_dir: str, files: list[Path], message: str) -> None:
    operations = [CommitOperationAdd(path_in_repo=f"{remote_dir}/{path.relative_to(output_root)}", path_or_fileobj=str(path)) for path in files if path.is_file()]
    if operations:
        api.create_commit(repo_id=RESULT_REPO, repo_type="dataset", operations=operations, commit_message=message)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN was not injected")
    self_tests()
    api = HfApi(token=token)
    reader = HubReader(token)
    # The organization-private source is mounted by the Job.  The OAuth token
    # used inside Jobs can read the user's private result repo but cannot query
    # this organization repo directly, so verify the mounted manifest by hash.
    manifest_path = DATASET_ROOT / "metadata" / "prompts.csv"
    if not manifest_path.is_file():
        raise RuntimeError("Mount the benchmark dataset at /dataset")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Mounted prompts.csv changed: "
            f"expected {EXPECTED_MANIFEST_SHA256}, found {manifest_sha256}"
        )

    source_root = f"benchmark_runs/{args.source_run_id}/{SOURCE_SHARD}"
    remote_dir = f"{source_root}/metrics/{args.metric_run_id}"
    output_root = Path("/tmp") / args.metric_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "metric_protocol.json"
    candidate_csv = output_root / "candidate_metrics.csv"
    prompt_csv = output_root / "prompt_metrics.csv"
    status_path = output_root / "status.json"

    existing_protocol = reader.get(RESULT_REPO, "dataset", "main", f"{remote_dir}/metric_protocol.json", optional=True)
    if existing_protocol and not args.force:
        protocol = json.loads(existing_protocol)
        source_revision = protocol["source_result_revision"]
    else:
        source_revision = api.dataset_info(RESULT_REPO).sha
        protocol = {
            "metric_protocol_id": "sam_audio_referring_all8_si_activity_rts_v1_20260903",
            "created_at": utc_now(), "source_result_repo": RESULT_REPO,
            "source_result_revision": source_revision, "source_run_id": args.source_run_id,
            "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "mounted_manifest_sha256": manifest_sha256,
            "candidate_count": EXPECTED_CANDIDATES, "target_sample_rate": TARGET_SR,
            "alignment": "independent zero-padded max-absolute FFT correlation within +/-50 ms",
            "si_sdr": "zero-mean scale-invariant SDR, report clipped to [-100,100] dB",
            "si_sdri": "aligned candidate SI-SDR minus independently aligned original-mixture SI-SDR",
            "activity": {
                "frame_ms": ACTIVITY_FRAME_MS, "hop_ms": ACTIVITY_HOP_MS,
                "threshold": "max(-60 dBFS, GT positive-frame RMS P95 - 35 dB)",
                "true_gt_primary": "strict frame-level F1",
                "near_gt_primary": "+/-250 ms collar-tolerant F1; <=300 ms isolated unannotated prediction bursts ignored only in tolerant score",
            },
            "rts": "10log10(mixture target-projected energy / model-residual target-projected energy) over GT-active frames",
            "rts_missing_policy": "unavailable; never substitute mixture minus generated target",
            "true_gt_datasets": sorted(TRUE_GT_DATASETS),
            "near_gt_datasets": sorted(NEAR_GT_DATASETS),
        }
        json_dump(protocol_path, protocol)

    # Resume the consolidated tables if this exact metric run already exists.
    if not args.force:
        for filename, destination in [("candidate_metrics.csv", candidate_csv), ("prompt_metrics.csv", prompt_csv)]:
            payload = reader.get(RESULT_REPO, "dataset", "main", f"{remote_dir}/{filename}", optional=True)
            if payload:
                destination.write_bytes(payload)
    candidate_rows: list[dict[str, Any]] = read_local_csv(candidate_csv)
    prompt_rows: list[dict[str, Any]] = read_local_csv(prompt_csv)
    completed = {int(row["row_index"]) for row in prompt_rows if row.get("status") == "success"}
    # A checkpoint can catch a prompt after some candidate rows were appended but
    # before its prompt-level success record was written.  Keep only atomically
    # completed prompts so a resume cannot duplicate partial candidate rows.
    prompt_rows = [row for row in prompt_rows if int(row["row_index"]) in completed]
    candidate_rows = [row for row in candidate_rows if int(row["row_index"]) in completed]
    if args.force:
        candidate_rows, prompt_rows, completed = [], [], set()

    manifest = read_csv_bytes(manifest_path.read_bytes())
    if len(manifest) != EXPECTED_PROMPTS:
        raise RuntimeError(f"Expected {EXPECTED_PROMPTS} manifest rows, found {len(manifest)}")
    selected = list(enumerate(manifest))[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    # One probe prevents 17,800 pointless 404 requests on the completed run.
    first_index = selected[0][0]
    residual_probe_path = f"{source_item_dir(args.source_run_id, first_index)}/candidates/candidate_01_residual.wav"
    residuals_available = reader.get(RESULT_REPO, "dataset", source_revision, residual_probe_path, optional=True) is not None
    protocol["source_model_residuals_available"] = residuals_available
    protocol["official_rts_available"] = residuals_available
    protocol["updated_at"] = utc_now()
    json_dump(protocol_path, protocol)

    print(f"source revision={source_revision}; selected={len(selected)}; resume={len(completed)}; residuals={residuals_available}", flush=True)
    cached_video_path = None
    cached_mixture = None
    cached_gt_path = None
    cached_reference = None
    cached_baseline = None
    processed_since_checkpoint = 0

    for position, (row_index, manifest_row) in enumerate(selected, start=1):
        if row_index in completed:
            continue
        started = time.monotonic()
        item_base = source_item_dir(args.source_run_id, row_index)
        prompt_record: dict[str, Any] = {
            "status": "error", "row_index": row_index, "dataset": manifest_row["dataset"],
            "domain": manifest_row.get("domain", ""), "task_type": manifest_row["task_type"],
            "clip_id": manifest_row["clip_id"], "target_source_id": manifest_row.get("target_source_id", ""),
            "prompt": manifest_row["prompt"], "error_type": "", "error": "",
        }
        current_candidates: list[dict[str, Any]] = []
        try:
            item_payload = reader.get(RESULT_REPO, "dataset", source_revision, f"{item_base}/item.json")
            assert item_payload is not None
            item = json.loads(item_payload)
            if item.get("status") != "success" or item.get("reranking_candidates") != EXPECTED_CANDIDATES:
                raise RuntimeError(f"Invalid source item status/config: {item.get('status')}")

            if cached_video_path != manifest_row["video_path"]:
                video_path = DATASET_ROOT / manifest_row["video_path"]
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                cached_mixture = decode_mixture_mp4(video_path.read_bytes())
                cached_video_path = manifest_row["video_path"]
                cached_gt_path = cached_reference = cached_baseline = None
            if cached_gt_path != manifest_row["gt_audio_path"]:
                gt_path = DATASET_ROOT / manifest_row["gt_audio_path"]
                if not gt_path.is_file():
                    raise FileNotFoundError(gt_path)
                cached_reference = wav_bytes_to_mono(gt_path.read_bytes())
                cached_gt_path = manifest_row["gt_audio_path"]
                mixture = fit_length(cached_mixture, len(cached_reference))
                aligned_mix, mix_lag, mix_corr = align_small_lag(cached_reference, mixture)
                cached_baseline = {
                    "mixture": mixture, "aligned": aligned_mix, "lag": mix_lag,
                    "corr": mix_corr, "si_sdr": standard_si_sdr(cached_reference, aligned_mix),
                }
                if residuals_available:
                    cached_baseline["rts_energy"], cached_baseline["rts_frames"] = target_projected_energy(cached_reference, aligned_mix)
            assert cached_reference is not None and cached_baseline is not None
            reference_tier = "true_gt" if manifest_row["dataset"] in TRUE_GT_DATASETS else "near_gt"

            for candidate_number in range(1, EXPECTED_CANDIDATES + 1):
                record: dict[str, Any] = {
                    "status": "error", "row_index": row_index, "candidate_number": candidate_number,
                    "dataset": manifest_row["dataset"], "domain": manifest_row.get("domain", ""),
                    "task_type": manifest_row["task_type"], "clip_id": manifest_row["clip_id"],
                    "video_path": manifest_row["video_path"], "gt_audio_path": manifest_row["gt_audio_path"],
                    "target_source_id": manifest_row.get("target_source_id", ""), "prompt": manifest_row["prompt"],
                    "reference_tier": reference_tier,
                    "visual_winner": candidate_number == int(item["visual_winner_candidate_number"]),
                    "text_winner": candidate_number == int(item["text_winner_candidate_number"]),
                    "fusion_winner": candidate_number == int(item["fusion_winner_candidate_number"]),
                    "error_type": "", "error": "",
                }
                try:
                    candidate_path = f"{item_base}/candidates/candidate_{candidate_number:02d}_target.wav"
                    target_payload = reader.get(RESULT_REPO, "dataset", source_revision, candidate_path)
                    assert target_payload is not None
                    prediction = fit_length(wav_bytes_to_mono(target_payload), len(cached_reference))
                    prediction_rms = rms(prediction)
                    gt_rms = rms(cached_reference)
                    prediction_vs_gt = db(prediction_rms / max(gt_rms, EPS))
                    silent = db(prediction_rms) < SILENT_ABSOLUTE_DBFS or prediction_vs_gt < SILENT_RELATIVE_TO_GT_DB
                    aligned, lag, corr_coeff = align_small_lag(cached_reference, prediction)
                    record.update(
                        status="success", prediction_sha256=hashlib.sha256(target_payload).hexdigest(),
                        prediction_rms_dbfs=db(prediction_rms), prediction_vs_gt_rms_db=prediction_vs_gt,
                        silent_failure=bool(silent), alignment_lag_samples=lag,
                        alignment_lag_ms=lag * 1000.0 / TARGET_SR, alignment_corr=corr_coeff,
                        si_sdr_db=standard_si_sdr(cached_reference, prediction),
                        aligned_si_sdr_db=standard_si_sdr(cached_reference, aligned),
                        mixture_alignment_lag_samples=cached_baseline["lag"],
                        mixture_alignment_lag_ms=cached_baseline["lag"] * 1000.0 / TARGET_SR,
                        mixture_alignment_corr=cached_baseline["corr"],
                        mixture_aligned_si_sdr_db=cached_baseline["si_sdr"],
                        si_sdri_db=standard_si_sdr(cached_reference, aligned) - cached_baseline["si_sdr"],
                    )
                    record.update(activity_metrics(cached_reference, aligned, reference_tier, bool(silent)))
                    if residuals_available:
                        residual_path = f"{item_base}/candidates/candidate_{candidate_number:02d}_residual.wav"
                        residual_payload = reader.get(RESULT_REPO, "dataset", source_revision, residual_path)
                        assert residual_payload is not None
                        residual = fit_length(wav_bytes_to_mono(residual_payload), len(cached_reference))
                        aligned_residual, _, _ = align_small_lag(cached_reference, residual)
                        residual_energy, active_frames = target_projected_energy(cached_reference, aligned_residual)
                        rts_db, rts_raw, remaining = rts_from_energies(cached_baseline["rts_energy"], residual_energy)
                        record.update(rts_status="success", rts_db=rts_db, rts_raw_db=rts_raw,
                                      rts_target_remaining_ratio=remaining, rts_target_remaining_percent=100 * remaining,
                                      rts_active_frames=active_frames)
                    else:
                        record.update(rts_status="unavailable_model_residual_not_saved", rts_db="", rts_raw_db="",
                                      rts_target_remaining_ratio="", rts_target_remaining_percent="", rts_active_frames="")
                except Exception as error:
                    record.update(error_type=type(error).__name__, error=str(error))
                current_candidates.append(record)
                candidate_rows.append(record)

            successful = [x for x in current_candidates if x["status"] == "success"]
            if len(successful) != EXPECTED_CANDIDATES:
                raise RuntimeError(f"Only {len(successful)}/8 candidates scored successfully")
            oracle = max(successful, key=lambda x: float(x["aligned_si_sdr_db"]))
            prompt_record.update(
                status="success", visual_winner_candidate_number=int(item["visual_winner_candidate_number"]),
                text_winner_candidate_number=int(item["text_winner_candidate_number"]),
                fusion_winner_candidate_number=int(item["fusion_winner_candidate_number"]),
                gt_oracle_candidate_number=int(oracle["candidate_number"]),
                gt_oracle_aligned_si_sdr_db=float(oracle["aligned_si_sdr_db"]),
                gt_oracle_si_sdri_db=float(oracle["si_sdri_db"]),
                candidate_success_count=len(successful), rts_available=residuals_available,
                elapsed_seconds=time.monotonic() - started,
            )
            for method, number in [
                ("visual", prompt_record["visual_winner_candidate_number"]),
                ("text", prompt_record["text_winner_candidate_number"]),
                ("fusion", prompt_record["fusion_winner_candidate_number"]),
            ]:
                chosen = next(x for x in successful if int(x["candidate_number"]) == int(number))
                prompt_record[f"{method}_aligned_si_sdr_db"] = chosen["aligned_si_sdr_db"]
                prompt_record[f"{method}_si_sdri_db"] = chosen["si_sdri_db"]
                prompt_record[f"{method}_activity_primary_f1"] = chosen["activity_primary_f1"]
                prompt_record[f"{method}_oracle_gap_db"] = float(oracle["aligned_si_sdr_db"]) - float(chosen["aligned_si_sdr_db"])
                prompt_record[f"{method}_oracle_exact_hit"] = int(number) == int(oracle["candidate_number"])
        except Exception as error:
            prompt_record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        prompt_rows.append(prompt_record)
        processed_since_checkpoint += 1
        print(f"[{position}/{len(selected)}] row={row_index} -> {prompt_record['status']} candidates={len(current_candidates)} elapsed={time.monotonic()-started:.1f}s", flush=True)

        if processed_since_checkpoint >= args.checkpoint_every:
            write_csv(candidate_csv, candidate_rows, CANDIDATE_FIELDS)
            prompt_fields = sorted({key for row in prompt_rows for key in row})
            write_csv(prompt_csv, prompt_rows, prompt_fields)
            json_dump(status_path, {"status": "running", "updated_at": utc_now(), "prompt_rows": len(prompt_rows),
                                    "successful_prompts": sum(row.get("status") == "success" for row in prompt_rows),
                                    "candidate_rows": len(candidate_rows), "source_result_revision": source_revision,
                                    "official_rts_available": residuals_available})
            if not args.no_upload:
                upload_checkpoint(api, output_root, remote_dir, [protocol_path, candidate_csv, prompt_csv, status_path],
                                  f"Checkpoint {args.metric_run_id}: {len(prompt_rows)} prompts")
            processed_since_checkpoint = 0

    write_csv(candidate_csv, candidate_rows, CANDIDATE_FIELDS)
    prompt_fields = sorted({key for row in prompt_rows for key in row})
    write_csv(prompt_csv, prompt_rows, prompt_fields)

    # One row per selector/prompt.  Oracle uses GT and is an analysis upper bound,
    # never a deployable SAM Audio result.
    candidates_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_prompt[int(row["row_index"])].append(row)
    selection_rows: list[dict[str, Any]] = []
    for prompt in prompt_rows:
        if prompt.get("status") != "success":
            continue
        candidates = candidates_by_prompt[int(prompt["row_index"])]
        for method, field in [
            ("visual", "visual_winner_candidate_number"),
            ("text", "text_winner_candidate_number"),
            ("fusion", "fusion_winner_candidate_number"),
            ("gt_oracle", "gt_oracle_candidate_number"),
        ]:
            selection_rows.append(selection_row(method, prompt, candidates, int(prompt[field])))
        # Fixed candidate-number summaries expose whether a particular draw is
        # systematically stronger; all eight underlying rows remain in the long
        # candidate table regardless of which reranker selected them.
        for candidate_number in range(1, EXPECTED_CANDIDATES + 1):
            selection_rows.append(
                selection_row(
                    f"candidate_{candidate_number:02d}",
                    prompt,
                    candidates,
                    candidate_number,
                )
            )
    selection_csv = output_root / "selection_metrics.csv"
    selection_fields = sorted({key for row in selection_rows for key in row})
    write_csv(selection_csv, selection_rows, selection_fields)

    aggregate_files: list[Path] = []
    for name, group_field in [("summary_overall.csv", None), ("summary_by_dataset.csv", "dataset"), ("summary_by_task_type.csv", "task_type")]:
        rows = build_aggregate_rows(selection_rows, group_field)
        path = output_root / name
        fields = sorted({key for row in rows for key in row})
        write_csv(path, rows, fields)
        aggregate_files.append(path)

    failures = [row for row in candidate_rows if row.get("status") != "success"]
    failures_path = output_root / "failures.csv"
    write_csv(failures_path, failures, CANDIDATE_FIELDS)
    final_status = {
        "status": "completed", "finished_at": utc_now(), "selected_prompt_count": len(selected),
        "recorded_prompt_count": len(prompt_rows),
        "successful_prompt_count": sum(row.get("status") == "success" for row in prompt_rows),
        "candidate_count": len(candidate_rows),
        "successful_candidate_count": sum(row.get("status") == "success" for row in candidate_rows),
        "candidate_failure_count": len(failures), "official_rts_available": residuals_available,
        "rts_unavailable_reason": None if residuals_available else "The completed generation run did not save model residual WAVs.",
        "source_result_revision": source_revision,
        "dataset_counts": dict(Counter(row.get("dataset") for row in prompt_rows if row.get("status") == "success")),
    }
    json_dump(status_path, final_status)
    readme = output_root / "README.md"
    readme.write_text(
        "# All-8 SAM Audio benchmark metrics\n\n"
        f"Source run: `{args.source_run_id}` at `{source_revision}`.\n\n"
        "`candidate_metrics.csv` contains one row for every prompt/candidate. "
        "`selection_metrics.csv` compares visual, text, fusion, and GT-oracle selection. "
        "The GT oracle is analysis-only.\n\n"
        + ("Official RTS was computed from saved model residuals.\n" if residuals_available else
           "Official RTS is marked unavailable because the source generation run did not save model residual WAVs. "
           "The evaluator intentionally does not replace residual with mixture minus target.\n"),
        encoding="utf-8",
    )
    final_files = [protocol_path, candidate_csv, prompt_csv, selection_csv, failures_path, status_path, readme, *aggregate_files]
    if not args.no_upload:
        upload_checkpoint(api, output_root, remote_dir, final_files, f"Complete {args.metric_run_id}")
    print(json.dumps(final_status, indent=2), flush=True)
    print(f"Results: https://huggingface.co/datasets/{RESULT_REPO}/tree/main/{remote_dir}", flush=True)


if __name__ == "__main__":
    main()
