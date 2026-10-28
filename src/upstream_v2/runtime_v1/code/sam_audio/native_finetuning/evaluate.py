"""Ground-truth validation for a native SAM Audio fine-tuning delta."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from sam_audio import SAMAudio, SAMAudioProcessor

from .data import NativeReferringDataset, _resolve, load_mono_audio
from .load import load_finetuned_sam_audio
from .train import _dtype, _resolve_dataset


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Return scale-invariant SDR in dB for one mono pair."""

    length = min(reference.numel(), estimate.numel())
    reference = reference.flatten()[:length].double()
    estimate = estimate.flatten()[:length].double()
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    reference_energy = reference.square().sum()
    if reference_energy <= 1.0e-12:
        return float("nan")
    projection = reference * ((estimate * reference).sum() / reference_energy)
    error = estimate - projection
    ratio = projection.square().sum() / error.square().sum().clamp_min(1.0e-12)
    return float(10.0 * torch.log10(ratio.clamp_min(1.0e-12)))


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value == value]
    return sum(finite) / len(finite) if finite else None


def _load_model(
    config: dict[str, Any],
    checkpoint: Path | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SAMAudio:
    model_id = config.get("model_local_path") or config.get("sam_audio_model", "facebook/sam-audio-large")
    if checkpoint is not None:
        return load_finetuned_sam_audio(
            checkpoint,
            model_id=model_id,
            revision=config.get("model_revision"),
            device=device,
            dtype=dtype,
            load_inference_helpers=False,
        )
    return SAMAudio.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        revision=config.get("model_revision"),
        visual_ranker=None,
        text_ranker=None,
        span_predictor=None,
        strict=True,
    ).to(device=device, dtype=dtype).eval()


def evaluate(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
    max_examples: int | None,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = _dtype(config.get("dtype", "bfloat16"))
    model = _load_model(
        config, checkpoint, device=device, dtype=dtype
    )
    processor = SAMAudioProcessor.from_pretrained(model._pretrained_model_path)
    data_root, manifest = _resolve_dataset(config)
    dataset = NativeReferringDataset(
        manifest,
        data_root=data_root,
        sample_rate=processor.audio_sampling_rate,
        max_seconds=config.get("max_seconds", 20.0),
        split="validation",
        validation_fraction=float(config.get("validation_fraction", 0.1)),
        split_seed=int(config.get("split_seed", 17)),
        split_group_column=config.get("split_group_column", "video_id"),
        split_stratify_column=config.get("split_stratify_column", "domain"),
        seed=seed,
        # Keep every target identity in selected validation clips so the
        # wrong-target comparison is complete. --max-examples limits costly
        # generation without deleting sibling GT stems from target-hit scoring.
        max_targets=None,
    )
    dataset.set_epoch(0)

    # Every same-clip stem is a possible identity. This makes target-hit rate
    # stricter than SI-SDR against the requested target alone.
    clip_targets: dict[tuple[str, str], dict[str, Path]] = {}
    for rows in dataset.examples:
        row = rows[0]
        clip_key = (row.get("dataset", ""), row["clip_id"])
        clip_targets.setdefault(clip_key, {})[row["target_source_id"]] = _resolve(
            data_root, row["gt_audio_path"]
        )

    count = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    records: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        mixture = item["mixture"]
        condition = processor(
            descriptions=[item["prompt"]],
            audios=[mixture],
            masked_videos=[item["video_path"]],
        ).to(device)
        torch.manual_seed(seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + index)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            prediction = model.separate(
                condition,
                reranking_candidates=1,
                predict_spans=False,
            ).target[0]
        prediction_cpu = prediction.detach().float().cpu()
        requested_score = si_sdr(item["target"], prediction_cpu)
        mixture_score = si_sdr(item["target"], mixture)

        clip_key = (item["dataset"], item["clip_id"])
        identity_scores: dict[str, float] = {}
        for target_id, target_path in clip_targets[clip_key].items():
            reference = load_mono_audio(target_path, processor.audio_sampling_rate)
            reference = reference[..., : mixture.size(-1)]
            identity_scores[target_id] = si_sdr(reference, prediction_cpu)
        best_target_id = max(identity_scores, key=identity_scores.get)
        wrong_scores = [
            score
            for target_id, score in identity_scores.items()
            if target_id != item["target_source_id"]
        ]
        best_wrong = max(wrong_scores) if wrong_scores else float("nan")
        records.append(
            {
                "sample_id": item["sample_id"],
                "dataset": item["dataset"],
                "domain": item["domain"],
                "clip_id": item["clip_id"],
                "target_source_id": item["target_source_id"],
                "prompt": item["prompt"],
                "si_sdr_db": requested_score,
                "mixture_si_sdr_db": mixture_score,
                "si_sdri_db": requested_score - mixture_score,
                "best_matching_target_id": best_target_id,
                "target_hit": int(best_target_id == item["target_source_id"]),
                "target_margin_db": (
                    requested_score - best_wrong if wrong_scores else float("nan")
                ),
                "same_clip_target_count": len(identity_scores),
            }
        )
        print(json.dumps({"event": "eval", "index": index, **records[-1]}), flush=True)

    if not records:
        raise RuntimeError("Validation split produced no examples")
    multi_target = [row for row in records if row["same_clip_target_count"] > 1]
    summary = {
        "checkpoint": str(checkpoint) if checkpoint is not None else "base",
        "examples": len(records),
        "multi_target_examples": len(multi_target),
        "mean_si_sdr_db": _mean([row["si_sdr_db"] for row in records]),
        "mean_si_sdri_db": _mean([row["si_sdri_db"] for row in records]),
        "target_hit_rate": _mean([row["target_hit"] for row in multi_target]),
        "mean_target_margin_db": _mean(
            [row["target_margin_db"] for row in multi_target]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_example.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluate(
        config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
