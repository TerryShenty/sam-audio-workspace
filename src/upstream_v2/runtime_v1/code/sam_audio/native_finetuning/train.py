"""Single-GPU native SAM Audio fine-tuning entry point.

This stage intentionally uses the released T5/video conditioning unchanged:
there is no VLM, special token, span prediction, candidate reranking, or
unpublished auxiliary AED loss in the training graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from sam_audio import SAMAudio, SAMAudioProcessor

from .data import NativeReferringDataset, collate_native_referring
from .checkpoint import verified_base, require_matching_base
from .model import (
    NativeSAMAudioFineTuner,
    configure_native_tuning,
    native_tuning_config_from_mapping,
    trainable_parameter_groups,
)


def _dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"dtype must be one of {sorted(values)}, got {name}")
    return values[name]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _stable_batch_seed(base_seed: int, sample_ids: list[str]) -> int:
    """Give a smoke-test batch the same RNG trajectory on every revisit."""

    payload = f"{base_seed}:" + "\x1f".join(sample_ids)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _resolve_dataset(config: dict[str, Any]) -> tuple[Path, Path]:
    if config.get("data_root"):
        root = Path(config["data_root"])
    else:
        repo_id = config["dataset_repo_id"]
        allow_patterns = config.get(
            "dataset_allow_patterns",
            ["training_data/**", "training_metadata/prompts.csv"],
        )
        if config.get("dataset_allow_patterns_file"):
            patterns_path = Path(config["dataset_allow_patterns_file"])
            allow_patterns = [
                line.strip()
                for line in patterns_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not allow_patterns:
                raise ValueError(f"No allow patterns in {patterns_path}")
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=config.get("dataset_revision", "training_set"),
                token=os.environ.get("HF_TOKEN"),
                allow_patterns=allow_patterns,
                cache_dir=config.get("dataset_cache_dir"),
            )
        )
    manifest = Path(config.get("manifest", "training_metadata/prompts.csv"))
    if not manifest.is_absolute():
        manifest = root / manifest
    if not manifest.exists():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")
    return root, manifest


def _make_dataset(
    config: dict[str, Any],
    *,
    split: str,
    data_root: Path,
    manifest: Path,
    sample_rate: int,
) -> NativeReferringDataset:
    max_key = "max_train_targets" if split == "train" else "max_validation_targets"
    dataset = NativeReferringDataset(
        manifest,
        data_root=data_root,
        sample_rate=sample_rate,
        max_seconds=config.get("max_seconds", 10.0),
        split=split,
        validation_fraction=float(config.get("validation_fraction", 0.1)),
        split_seed=int(config.get("split_seed", 17)),
        split_group_column=config.get("split_group_column", "video_id"),
        split_stratify_column=config.get("split_stratify_column", "domain"),
        seed=int(config.get("seed", 0)),
        max_targets=config.get(max_key),
    )
    dataset.target_only = bool(config.get("target_only", False))
    dataset.require_explicit_mixture_path = bool(config.get("require_explicit_mixture_path", False))
    return dataset


def _make_condition_batch(
    processor: SAMAudioProcessor,
    batch: dict[str, Any],
    device: torch.device,
):
    audios = [
        batch["mixture"][index, :, : int(size)]
        for index, size in enumerate(batch["audio_sizes"])
    ]
    return processor(
        descriptions=batch["prompt"],
        audios=audios,
        # Passing the original video here is intentional. The public API calls
        # this argument masked_videos, but no spatial mask is applied.
        masked_videos=batch["video_path"],
    ).to(device)


def _parameter_count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _save_checkpoint(
    path: Path,
    *,
    model: NativeSAMAudioFineTuner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    tuning_summary: dict[str, Any],
    global_step: int,
    epoch: int,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "sam_audio_native_referring_v1",
            "verified_base": verified_base(model.sam_audio),
            "global_step": global_step,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "config": config,
            "tuning_summary": tuning_summary,
            "trainable_model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _load_resume(
    path: Path,
    *,
    model: NativeSAMAudioFineTuner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require_matching_base(checkpoint, model.sam_audio)
    if checkpoint.get("format") != "sam_audio_native_referring_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint["global_step"]),
        int(checkpoint["epoch"]),
        float(checkpoint.get("best_validation_loss", float("inf"))),
    )


@torch.no_grad()
def _validate(
    model: NativeSAMAudioFineTuner,
    processor: SAMAudioProcessor,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    max_batches: int | None,
) -> float:
    model.eval()
    losses: list[float] = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    # Resetting both time and Gaussian noise makes validation FM loss comparable
    # across checkpoints instead of being dominated by Monte-Carlo variation.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            condition_batch = _make_condition_batch(processor, batch, device)
            target = batch["target"].to(device, non_blocking=True)
            residual = batch["residual"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and amp_dtype != torch.float32,
            ):
                output = model(
                    condition_batch=condition_batch,
                    target=target,
                    residual=residual,
                )
            losses.append(float(output.loss))
            del condition_batch, target, residual, output
    model.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return sum(losses) / len(losses)


def train(config: dict[str, Any]) -> None:
    if config.get("target_only", False):
        raise ValueError(
            "Target-only training requires sam_audio.native_finetuning.optimized_train; "
            "this legacy joint runner requires residual supervision"
        )
    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_dtype = _dtype(config.get("dtype", "bfloat16"))
    output_dir = Path(config.get("output_dir", "/outputs/native_referring_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", config)
    metrics_path = output_dir / "metrics.jsonl"
    if not config.get("resume_from"):
        metrics_path.write_text("", encoding="utf-8")

    def emit(record: dict[str, Any]) -> None:
        print(json.dumps(record, ensure_ascii=False), flush=True)
        _append_jsonl(metrics_path, record)

    model_id = config.get("model_local_path") or config.get("sam_audio_model", "facebook/sam-audio-large")
    # Rankers and PEA-Frame are inference helpers and do not belong in this
    # training graph. Strict loading permits only documented helper omissions.
    sam_audio = SAMAudio.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        revision=config.get("model_revision"),
        visual_ranker=None,
        text_ranker=None,
        span_predictor=None,
        strict=True,
    )
    processor = SAMAudioProcessor.from_pretrained(sam_audio._pretrained_model_path)
    _write_json(output_dir / "pretrained-load.json", sam_audio._pretrained_load_report)
    sam_audio.to(device=device, dtype=amp_dtype)

    tuning_config = native_tuning_config_from_mapping(config)
    tuning_summary = configure_native_tuning(sam_audio, tuning_config)
    _write_json(output_dir / "tuning_summary.json", tuning_summary)
    emit({"event": "tuning_summary", **tuning_summary})

    model = NativeSAMAudioFineTuner(
        sam_audio,
        sigma_min=float(config.get("sigma_min", 1.0e-5)),
    ).to(device)
    model.train()

    data_root, manifest = _resolve_dataset(config)
    train_dataset = _make_dataset(
        config,
        split="train",
        data_root=data_root,
        manifest=manifest,
        sample_rate=processor.audio_sampling_rate,
    )
    validation_dataset = _make_dataset(
        config,
        split="validation",
        data_root=data_root,
        manifest=manifest,
        sample_rate=processor.audio_sampling_rate,
    )
    batch_size = int(config.get("batch_size", 1))
    workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        # Workers are recreated every epoch so dataset.set_epoch() changes the
        # sampled paraphrase in their dataset copies.
        persistent_workers=False,
        collate_fn=collate_native_referring,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, min(workers, 2)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_native_referring,
    )

    parameter_groups = trainable_parameter_groups(
        sam_audio,
        lora_learning_rate=float(config.get("lora_learning_rate", 1.0e-4)),
        condition_learning_rate=float(
            config.get("condition_learning_rate", 1.0e-5)
        ),
        block_learning_rate=float(config.get("block_learning_rate", 5.0e-6)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(config.get("adam_betas", [0.9, 0.95])),
        eps=float(config.get("adam_epsilon", 1.0e-8)),
    )
    grad_accumulation = int(config.get("gradient_accumulation_steps", 8))
    num_epochs = int(config.get("num_epochs", 3))
    updates_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
    configured_max_steps = config.get("max_steps")
    max_steps = (
        num_epochs * updates_per_epoch
        if configured_max_steps is None
        else int(configured_max_steps)
    )
    warmup_steps = int(config.get("warmup_steps", min(500, max_steps // 10)))
    scheduler_kind = config.get("scheduler", "constant")

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        if scheduler_kind == "constant":
            return 1.0
        if scheduler_kind == "cosine":
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        raise ValueError("scheduler must be constant or cosine")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    global_step = 0
    start_epoch = 0
    best_validation_loss = float("inf")
    if config.get("resume_from"):
        global_step, start_epoch, best_validation_loss = _load_resume(
            Path(config["resume_from"]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    hardware = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / 2**30,
                "gpu_compute_capability": [properties.major, properties.minor],
            }
        )
        torch.cuda.reset_peak_memory_stats(device)
    emit(
        {
            "event": "data",
            **hardware,
            "data_root": str(data_root),
            "manifest": str(manifest),
            "train_targets": len(train_dataset),
            "validation_targets": len(validation_dataset),
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accumulation,
            "max_steps": max_steps,
            "fixed_train_rng_seed": config.get("fixed_train_rng_seed"),
            "optimizer_groups": {
                group["group_name"]: _parameter_count(group["params"])
                for group in parameter_groups
            },
        }
    )

    # A real throughput benchmark must exclude model startup, dataloader
    # worker startup, validation, and checkpoint writes.  The smoke test did
    # not do this, so its end-to-end Job duration cannot be extrapolated to a
    # 30k-video run.  These counters measure target examples processed by the
    # actual training loop after a configurable warmup.
    benchmark_warmup_updates = max(
        0, int(config.get("benchmark_warmup_updates", 0))
    )
    profile_stage_updates = max(
        0, int(config.get("profile_stage_updates", 0))
    )
    benchmark_started_at: float | None = None
    benchmark_excluded_seconds = 0.0
    benchmark_target_examples = 0
    benchmark_audio_seconds = 0.0
    benchmark_micro_batches = 0
    benchmark_optimizer_updates = 0
    stage_seconds = {
        "data_wait": 0.0,
        "condition_and_transfer": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    stage_counts = {name: 0 for name in stage_seconds}

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if bool(config.get("validate_before_training", False)):
        initial_validation_loss = _validate(
            model,
            processor,
            validation_loader,
            device=device,
            amp_dtype=amp_dtype,
            seed=int(config.get("validation_seed", 2026)),
            max_batches=config.get("validation_max_batches", 64),
        )
        emit(
            {
                "event": "validation",
                "step": 0,
                "flow_loss": initial_validation_loss,
            }
        )

    if benchmark_warmup_updates == 0:
        synchronize()
        benchmark_started_at = time.perf_counter()

    log_every = int(config.get("log_every", 10))
    save_every = int(config.get("save_every", 500))
    validate_every = int(config.get("validate_every", 500))
    validation_max_batches = config.get("validation_max_batches", 64)
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    trainable_parameters = [
        parameter for parameter in sam_audio.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)

    accumulation_count = 0
    running_loss = 0.0
    running_micro_batches = 0
    epoch = start_epoch
    while global_step < max_steps and epoch < num_epochs:
        train_dataset.set_epoch(epoch)
        next_batch_requested_at = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch_fetch_seconds = time.perf_counter() - next_batch_requested_at
            profile_this_batch = global_step < profile_stage_updates
            if profile_this_batch:
                stage_seconds["data_wait"] += batch_fetch_seconds
                stage_counts["data_wait"] += 1
            # This is useful only for an overfit smoke test. Reusing the same
            # flow time/noise/dropout path makes loss trend interpretable on a
            # tiny repeated sample. Normal training leaves the option unset.
            if config.get("fixed_train_rng_seed") is not None:
                batch_seed = _stable_batch_seed(
                    int(config["fixed_train_rng_seed"]), batch["sample_id"]
                )
                torch.manual_seed(batch_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(batch_seed)

            if profile_this_batch:
                synchronize()
                stage_started_at = time.perf_counter()
            condition_batch = _make_condition_batch(processor, batch, device)
            target = batch["target"].to(device, non_blocking=True)
            residual = batch["residual"].to(device, non_blocking=True)
            if profile_this_batch:
                synchronize()
                stage_seconds["condition_and_transfer"] += (
                    time.perf_counter() - stage_started_at
                )
                stage_counts["condition_and_transfer"] += 1

            if profile_this_batch:
                synchronize()
                stage_started_at = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and amp_dtype != torch.float32,
            ):
                output = model(
                    condition_batch=condition_batch,
                    target=target,
                    residual=residual,
                )
                scaled_loss = output.loss / grad_accumulation
            if profile_this_batch:
                synchronize()
                stage_seconds["forward"] += time.perf_counter() - stage_started_at
                stage_counts["forward"] += 1

            if profile_this_batch:
                synchronize()
                stage_started_at = time.perf_counter()
            scaled_loss.backward()
            if profile_this_batch:
                synchronize()
                stage_seconds["backward"] += time.perf_counter() - stage_started_at
                stage_counts["backward"] += 1

            if benchmark_started_at is not None:
                benchmark_target_examples += len(batch["sample_id"])
                benchmark_audio_seconds += float(batch["audio_sizes"].sum()) / float(
                    processor.audio_sampling_rate
                )
                benchmark_micro_batches += 1
            running_loss += float(output.loss.detach())
            running_micro_batches += 1
            accumulation_count += 1
            last_batch = batch_index + 1 == len(train_loader)
            if accumulation_count < grad_accumulation and not last_batch:
                del condition_batch, target, residual, output, scaled_loss
                next_batch_requested_at = time.perf_counter()
                continue

            if accumulation_count < grad_accumulation:
                correction = grad_accumulation / accumulation_count
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if profile_this_batch:
                synchronize()
                stage_started_at = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if profile_this_batch:
                synchronize()
                stage_seconds["optimizer"] += time.perf_counter() - stage_started_at
                stage_counts["optimizer"] += 1
            accumulation_count = 0
            global_step += 1

            if benchmark_started_at is not None:
                benchmark_optimizer_updates += 1
            elif global_step >= benchmark_warmup_updates:
                synchronize()
                benchmark_started_at = time.perf_counter()

            if global_step == 1 or global_step % log_every == 0:
                record = {
                    "event": "train",
                    "step": global_step,
                    "epoch": epoch,
                    # Average the unscaled objective over micro-batches. Dividing
                    # by optimizer updates would overstate this by the gradient
                    # accumulation factor.
                    "loss": running_loss / max(running_micro_batches, 1),
                    "last_loss": float(output.loss.detach()),
                    "prediction_rms": float(output.prediction_rms),
                    "target_velocity_rms": float(output.target_velocity_rms),
                    "grad_norm": float(grad_norm),
                    "learning_rates": {
                        group["group_name"]: group["lr"]
                        for group in optimizer.param_groups
                    },
                    "gpu_max_memory_gb": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                if device.type == "cuda":
                    record.update(
                        {
                            "gpu_current_allocated_gb": torch.cuda.memory_allocated(
                                device
                            )
                            / 2**30,
                            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved(device)
                            / 2**30,
                        }
                    )
                emit(record)
                running_loss = 0.0
                running_micro_batches = 0

            if validate_every and global_step % validate_every == 0:
                if benchmark_started_at is not None:
                    synchronize()
                    excluded_started_at = time.perf_counter()
                validation_loss = _validate(
                    model,
                    processor,
                    validation_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    seed=int(config.get("validation_seed", 2026)),
                    max_batches=validation_max_batches,
                )
                if benchmark_started_at is not None:
                    synchronize()
                    benchmark_excluded_seconds += (
                        time.perf_counter() - excluded_started_at
                    )
                emit(
                    {
                        "event": "validation",
                        "step": global_step,
                        "flow_loss": validation_loss,
                    }
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    if benchmark_started_at is not None:
                        synchronize()
                        excluded_started_at = time.perf_counter()
                    _save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        tuning_summary=tuning_summary,
                        global_step=global_step,
                        epoch=epoch,
                        best_validation_loss=best_validation_loss,
                    )
                    if benchmark_started_at is not None:
                        synchronize()
                        benchmark_excluded_seconds += (
                            time.perf_counter() - excluded_started_at
                        )

            if save_every and global_step % save_every == 0:
                if benchmark_started_at is not None:
                    synchronize()
                    excluded_started_at = time.perf_counter()
                _save_checkpoint(
                    output_dir / f"step_{global_step:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    tuning_summary=tuning_summary,
                    global_step=global_step,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )
                if benchmark_started_at is not None:
                    synchronize()
                    benchmark_excluded_seconds += (
                        time.perf_counter() - excluded_started_at
                    )
            del condition_batch, target, residual, output, scaled_loss
            if global_step >= max_steps:
                break
            next_batch_requested_at = time.perf_counter()
        epoch += 1

    if benchmark_started_at is not None and benchmark_target_examples:
        synchronize()
        benchmark_wall_seconds = time.perf_counter() - benchmark_started_at
        benchmark_training_seconds = max(
            benchmark_wall_seconds - benchmark_excluded_seconds, 1.0e-9
        )
        targets_per_second = benchmark_target_examples / benchmark_training_seconds
        estimate_targets_per_epoch = config.get(
            "benchmark_estimate_targets_per_epoch"
        )
        estimated_hours: dict[str, float] = {}
        if estimate_targets_per_epoch is not None:
            estimate_targets_per_epoch = int(estimate_targets_per_epoch)
            for epochs in config.get("benchmark_estimate_epochs", [1, 2]):
                epochs = int(epochs)
                estimated_hours[f"{epochs}_epoch"] = (
                    estimate_targets_per_epoch * epochs / targets_per_second / 3600.0
                )
        emit(
            {
                "event": "throughput_summary",
                "warmup_optimizer_updates": benchmark_warmup_updates,
                "timed_optimizer_updates": benchmark_optimizer_updates,
                "timed_micro_batches": benchmark_micro_batches,
                "timed_target_examples": benchmark_target_examples,
                "timed_audio_seconds": benchmark_audio_seconds,
                "wall_seconds": benchmark_wall_seconds,
                "excluded_validation_and_checkpoint_seconds": (
                    benchmark_excluded_seconds
                ),
                "training_seconds": benchmark_training_seconds,
                "targets_per_second": targets_per_second,
                "seconds_per_target": 1.0 / targets_per_second,
                "audio_seconds_per_wall_second": (
                    benchmark_audio_seconds / benchmark_training_seconds
                ),
                "estimated_hours": estimated_hours,
                "stage_profile_updates": profile_stage_updates,
                "stage_average_seconds": {
                    name: (
                        stage_seconds[name] / stage_counts[name]
                        if stage_counts[name]
                        else None
                    )
                    for name in stage_seconds
                },
                "gpu_max_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
                "gpu_max_reserved_gb": (
                    torch.cuda.max_memory_reserved(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
        )

    _save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        tuning_summary=tuning_summary,
        global_step=global_step,
        epoch=epoch,
        best_validation_loss=best_validation_loss,
    )
    completion = {
        "event": "complete",
        "step": global_step,
        "best_validation_loss": best_validation_loss,
        "output_dir": str(output_dir),
    }
    if device.type == "cuda":
        completion.update(
            {
                "gpu_max_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
            }
        )
    emit(completion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    train(config)


if __name__ == "__main__":
    main()
