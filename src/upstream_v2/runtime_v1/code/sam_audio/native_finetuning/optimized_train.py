"""Local-only prepare/profile/train CLI for the native SAM Audio adaptation.

No HF upload, job submission, reranking, span prediction, VLM, or AED objective.
Use torchrun for DDP; plain python for one GPU. See docs/native_optimized_training.md.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .benchmark_utils import collate_features, ddp_weighted_loss, per_target_mse
from .feature_cache import (
    CachedTrainingDataset,
    ComponentStore,
    FrozenExtractor,
    build_components,
    collate_cached_training,
    finalize_catalog,
    make_cache_plan,
)
from .objective import make_flow_matching_state
from .checkpoint import verified_base, require_matching_base
from .preflight import audit_training_separation
from .optimized_utils import (
    LengthBucketPlan,
    StageTimer,
    atomic_json,
    device_memory,
    digest,
    move_features,
    throughput_summary,
)


class CachedFlowModel(torch.nn.Module):
    def __init__(self, sam_audio, sigma_min=1e-5):
        super().__init__()
        self.sam_audio, self.sigma_min = sam_audio, sigma_min
        self.referring_aux = None
        self.last_loss_terms = None
        self.ema = None

    def train(self, mode=True):
        super().train(mode)
        self.sam_audio.eval()
        for name in ("transformer", "memory_proj", "align_masked_video"):
            module = getattr(self.sam_audio, name, None)
            if module is not None and (
                name == "transformer"
                or any(p.requires_grad for p in module.parameters())
            ):
                module.train(mode)
        return self

    def forward(self, features, *, time=None, noise=None):
        state = make_flow_matching_state(
            features["clean"], sigma_min=self.sigma_min, time=time, noise=noise
        )
        use_referring = self.referring_aux is not None and "referring_negatives" in features
        extra = {"return_hidden_layer": self.referring_aux.layer} if use_referring else {}
        prediction = self.sam_audio(
            noisy_audio=state.noisy_latent, time=state.time, **features["forward_args"], **extra
        )
        if use_referring:
            prediction, hidden = prediction
        fm = per_target_mse(
            prediction,
            state.target_velocity,
            features["forward_args"]["audio_pad_mask"],
        )

        referring = torch.zeros_like(fm)
        if use_referring:
            referring = self.referring_aux(hidden, features["clean"][..., :128],
                features["referring_negatives"], features["referring_negative_valid"],
                features["forward_args"]["audio_pad_mask"])
        self.last_loss_terms = (fm.detach(), referring.detach())
        return fm + referring if use_referring else fm

    def trainable_state_dict(self):
        return {
            n: p.detach().cpu()
            for n, p in self.sam_audio.named_parameters()
            if p.requires_grad
        }


def release_frozen_encoders(sam):
    """Detach unused modules BEFORE constructing DDP (no mixed CPU/GPU module).

    Only the cached training wrapper may use this stripped instance. Reload the
    normal base+delta for inference; the public model files are not modified.
    """
    for name in ("audio_codec", "text_encoder", "vision_encoder"):
        module = getattr(sam, name)
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError(f"Cannot release trainable {name}")
        setattr(sam, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_cached_example(wrapper, fresh, cached, *, device, check_gradients=True):
    """Independent fresh official encoder output vs cache; no optimizer update."""
    for key, value in fresh["forward_args"].items():
        torch.testing.assert_close(
            value.cpu(), cached["forward_args"][key], rtol=1e-3, atol=1e-4,
            msg=lambda message, key=key: f"Cached feature {key}: {message}",
        )
    torch.testing.assert_close(
        fresh["clean"].cpu(), cached["clean"], rtol=1e-3, atol=1e-4
    )
    # Seed dropout/noise identically for both paths and restore the caller RNG.
    params = [(n, p) for n, p in wrapper.named_parameters() if p.requires_grad]
    cuda_devices = (
        [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    )
    previous_mode = wrapper.training
    wrapper.train()
    reference = None
    try:
        for features in (fresh, cached):
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(1729)
                wrapper.zero_grad(set_to_none=True)
                losses = wrapper(move_features(features, device))
                gradients = {}
                if check_gradients:
                    losses.mean().backward()
                    gradients = {
                        n: None if p.grad is None else p.grad.detach().cpu().clone()
                        for n, p in params
                    }
                if reference is None:
                    reference = (losses.detach().cpu(), gradients)
                else:
                    torch.testing.assert_close(
                        losses.detach().cpu(), reference[0], rtol=1e-3, atol=1e-4
                    )
                    for name, gradient in gradients.items():
                        before = reference[1][name]
                        if gradient is None or before is None:
                            if gradient is not before:
                                raise AssertionError(
                                    f"Gradient presence changed: {name}"
                                )
                        else:
                            torch.testing.assert_close(
                                gradient, before, rtol=2e-2, atol=1e-4
                            )
    finally:
        wrapper.zero_grad(set_to_none=True)
        wrapper.train(previous_mode)
    return {
        "features": "passed",
        "loss": "passed",
        "gradients": "passed" if check_gradients else "not_checked",
    }


def gather_objects(value, world):
    if world == 1:
        return [value]
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def training_splits(config):
    """A zero fraction explicitly disables held-out validation, not training."""
    fraction = float(config.get("validation_fraction", 0.1))
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    return ("train", "validation") if fraction else ("train",)


def model_signature(model_path, config, processor, sam):
    """Identity includes loaded base weights, preprocessing source and runtime."""
    import importlib.metadata

    root = Path(__file__).parents[1]
    code = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root / "processor.py",
            root / "model/codec.py",
            root / "model/vision_encoder.py",
            root / "model/text_encoder.py",
            root / "model/base.py",
            root / "model/model.py",
            root / "native_finetuning/checkpoint.py",
            root / "native_finetuning/data.py",
            Path(__file__),
            root / "native_finetuning/feature_cache.py",
            root / "native_finetuning/target_only.py",
        ]
    }
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "torchaudio",
        "torchcodec",
        "transformers",
        "dacvae",
        "perception-models",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    # HF snapshots have an immutable commit directory. For a custom local model
    # record checkpoint stat; changing that file creates a new namespace.
    weights = Path(model_path) / "checkpoint.pt"
    stat = weights.stat()
    resolved = Path(model_path).resolve()
    if resolved.parent.name == "snapshots" and len(resolved.name) == 40:
        base_identity = {
            "hub_model": config["sam_audio_model"],
            "revision": resolved.name,
        }
    else:
        base_identity = {
            "local_model_path": str(resolved),
            "checkpoint_mtime_ns": stat.st_mtime_ns,
        }
    return {
        "version": 2,
        "verified_base": verified_base(sam),
        "base_identity": base_identity,
        "checkpoint_bytes": stat.st_size,
        "model_config_sha256": hashlib.sha256(
            (Path(model_path) / "config.json").read_bytes()
        ).hexdigest(),
        "dtype": config["dtype"],
        "sample_rate": processor.audio_sampling_rate,
        "hop_length": processor.audio_hop_length,
        "max_seconds": config.get("max_seconds"),
        "allow_tf32": config.get("allow_tf32", True),
        "video_frame_chunk": int(config.get("video_frame_chunk", 0)),
        "sources": code,
        "versions": versions,
    }


def save_checkpoint(
    path,
    wrapper,
    optimizer,
    scheduler,
    config,
    tuning,
    *,
    step,
    next_epoch,
    next_update,
    world,
    device,
    data_identity,
):
    rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu()
        if device.type == "cuda"
        else None,
    }
    states = gather_objects(rng, world)
    if not dist.is_initialized() or dist.get_rank() == 0:
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "format": "sam_audio_native_referring_v1",
                "verified_base": verified_base(wrapper.sam_audio),
                "optimized_recipe": 1,
                "global_step": step,
                "epoch": next_epoch,
                "next_update": next_update,
                "world_size": world,
                "rng_by_rank": states,
                "data_identity": data_identity,
                "config": config,
                "tuning_summary": tuning,
                "trainable_model": wrapper.trainable_state_dict(),
                "referring_aux": wrapper.referring_aux.state_dict() if wrapper.referring_aux is not None else None,
                "ema": wrapper.ema.state_dict() if wrapper.ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            temporary,
        )
        temporary.replace(path)
        atomic_json(
            path.parent / "latest-checkpoint.json",
            {"path": path.name, "global_step": step,
             "next_epoch": next_epoch, "next_update": next_update},
        )


def prune_step_checkpoints(output, keep):
    """Bound rolling saves in this new run directory; preserve epoch/final saves."""
    if keep <= 0:
        return
    paths = sorted(
        (p for p in output.glob("step-*.pt") if p.stem[5:].isdigit()),
        key=lambda p: int(p.stem[5:]),
    )
    for path in paths[:-keep]:
        path.unlink()


@torch.no_grad()
def fixed_flow_probe(
    wrapper,
    catalog,
    store,
    config,
    device,
    dtype,
    rank,
    world,
    *,
    split,
    max_targets,
    seed,
):
    """Measure the same examples at the same flow time/noise on every call.

    Validation remains held out.  A small optional train-split probe is useful
    only as an optimizer/DDP smoke gate: it answers whether the selected
    trainable parameters can reduce the exact objective they were optimized
    on, not whether separation quality or generalization improved.
    """
    entries = catalog["splits"][split]["entries"]
    limit = min(len(entries), int(max_targets))
    total = torch.zeros(2, device=device, dtype=torch.float64)
    was_training = wrapper.training
    wrapper.eval()
    with torch.random.fork_rng(
        devices=[device.index or 0] if device.type == "cuda" else []
    ):
        for index in range(rank, limit, world):
            entry = entries[index]
            torch.manual_seed(
                int(
                    digest([seed, split, entry["sample_id"]])[:12],
                    16,
                )
            )
            value = move_features(store.assemble(entry["variants"][0]), device)
            with torch.autocast(
                device.type, dtype=dtype, enabled=device.type == "cuda"
            ):
                loss = wrapper(value).sum()
            total[0] += loss.double()
            total[1] += 1
    if world > 1:
        dist.all_reduce(total)
    wrapper.train(was_training)
    return {
        "flow_mse": (total[0] / total[1].clamp_min(1)).item(),
        "targets": int(total[1].item()),
        "split": split,
        "note": "Fixed-noise/time velocity loss, not separated-audio quality",
    }


def validate(wrapper, catalog, store, config, device, dtype, rank, world):
    return fixed_flow_probe(
        wrapper,
        catalog,
        store,
        config,
        device,
        dtype,
        rank,
        world,
        split="validation",
        max_targets=int(config.get("validation_max_targets", 128)),
        seed=int(config.get("validation_seed", 2026)),
    )


def training_probe(wrapper, catalog, store, config, device, dtype, rank, world):
    limit = int(config.get("training_probe_max_targets", 0))
    if limit <= 0:
        return None
    return fixed_flow_probe(
        wrapper,
        catalog,
        store,
        config,
        device,
        dtype,
        rank,
        world,
        split="train",
        max_targets=limit,
        seed=int(
            config.get("training_probe_seed", config.get("validation_seed", 2026))
        ),
    )


def run_training(
    config,
    *,
    mode,
    feature_mode,
    wrapper,
    extractor,
    catalog,
    store,
    native_dataset,
    tuning,
    output,
    device,
    dtype,
    rank,
    world,
):
    from .model import trainable_parameter_groups

    timer = extractor.timer
    validation_targets = len(catalog["splits"].get("validation", {}).get("entries", []))
    has_validation = validation_targets > 0
    diagnostic = mode == "profile"
    if diagnostic and config.get("compile", False):
        raise ValueError(
            "Disable compile for instrumented profile; benchmark compile separately"
        )
    cached = feature_mode == "cached"
    referring_config = config.get("referring_contrastive") or {}
    referring_pairs = None
    if referring_config:
        from .referring_aux import ReferringAux, ReviewedPairs
        if not cached:
            raise ValueError("Referring auxiliary currently requires cached training")
        referring_pairs = ReviewedPairs(referring_config["pairs_path"], catalog["splits"]["train"]["entries"])
        referring_config = dict(referring_config)
        referring_config["pairs_sha256"] = referring_pairs.sha256
        config["referring_contrastive"] = referring_config
        layer = int(referring_config.get("layer", 5))
        if not 0 <= layer < len(wrapper.sam_audio.transformer.layers):
            raise ValueError("Invalid referring hidden layer")
        wrapper.referring_aux = ReferringAux(
            wrapper.sam_audio.transformer.norm.weight.numel(), layer=layer,
            weight=float(referring_config.get("weight", 0.1)),
            temperature=float(referring_config.get("temperature", 0.07)),
            head_dim=int(referring_config.get("head_dim", 2048))).to(device)
        atomic_json(output / f"referring-pairs-rank{rank}.json", {
            "sha256": referring_pairs.sha256, "reviewed_prompt_pairs": len(referring_pairs.pairs),
            "teacher": "cached DAC target first 128 channels; not official AED",
            "normalization": "all real examples; unreviewed prompts contribute zero auxiliary"})
    if cached:
        release_frozen_encoders(wrapper.sam_audio)
    wrapper.train()
    # Keep parameter references to the original module, independent of compile/DDP.
    groups = trainable_parameter_groups(
        wrapper.sam_audio,
        lora_learning_rate=float(config["lora_learning_rate"]),
        condition_learning_rate=float(config["condition_learning_rate"]),
        block_learning_rate=float(config["block_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    if wrapper.referring_aux is not None:
        groups.append({"params": list(wrapper.referring_aux.parameters()),
            "lr": float(referring_config.get("learning_rate", config["condition_learning_rate"])),
            "weight_decay": float(config["weight_decay"])})
    parameters = [p for g in groups for p in g["params"]]
    if config.get("ema_decay") is not None:
        from .ema import TrainableEMA
        wrapper.ema = TrainableEMA(wrapper, float(config["ema_decay"]))
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(config["adam_betas"]),
        eps=float(config["adam_epsilon"]),
        fused=bool(config.get("fused_adamw", True)) and device.type == "cuda",
    )
    dataset = CachedTrainingDataset(
        catalog, "train", store, online_dataset=None if cached else native_dataset,
        referring_pairs=referring_pairs
    )
    plan = LengthBucketPlan(
        [v["latent_frames"] for v in dataset.entries],
        world_size=world,
        rank=rank,
        micro_batch=int(config["batch_size"]),
        accumulation=int(config["gradient_accumulation_steps"]),
        seed=int(config["seed"]),
        bucket_batches=int(config.get("bucket_batches", 32)),
        exact_lengths=bool(config.get("exact_lengths", True)),
    )
    workers = int(config.get("num_workers", 4))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers:
        loader_kwargs.update(
            prefetch_factor=int(config.get("prefetch_factor", 2)),
            multiprocessing_context="spawn",
        )
    # Explicit epoch tokens make persistent workers safe (no stale epoch copies).
    loader = DataLoader(
        dataset,
        batch_sampler=plan,
        collate_fn=collate_cached_training,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
        **loader_kwargs,
    )
    epochs = int(config["num_epochs"])
    max_steps = int(config.get("max_steps") or epochs * plan.updates_per_epoch)
    if diagnostic:
        max_steps = min(max_steps, int(config.get("profile_updates", 12)))
    warmup = int(config.get("warmup_steps", 0))

    def schedule(step):
        if warmup and step < warmup:
            return max(1, step) / warmup
        if config.get("scheduler", "constant") == "cosine":
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
                )
            )
        if config.get("scheduler", "constant") != "constant":
            raise ValueError("scheduler must be constant or cosine")
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    identity = {"plan_id": catalog["plan_id"], "signature": catalog["signature"]}
    step, start_epoch, skip_updates = 0, 0, 0
    resume = config.get("resume_from")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        require_matching_base(checkpoint, wrapper.sam_audio)
        if (
            checkpoint.get("optimized_recipe") != 1
            or checkpoint["data_identity"] != identity
        ):
            raise ValueError(
                "Resume requires this recipe and exactly matching data/cache/base identity"
            )
        if checkpoint["world_size"] != world:
            raise ValueError("Exact resume requires the same world size")
        if checkpoint["tuning_summary"]["config"] != tuning["config"]:
            raise ValueError("Resume tuning configuration changed")
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "bucket_batches",
            "exact_lengths",
            "seed",
            "num_epochs",
            "max_steps",
            "warmup_steps",
            "scheduler",
            "weight_decay",
            "adam_betas",
            "adam_epsilon",
            "max_grad_norm",
            "lora_learning_rate",
            "condition_learning_rate",
            "block_learning_rate",
            "sigma_min",
            "referring_contrastive",
            "ema_decay",
        ):
            if checkpoint["config"].get(key) != config.get(key):
                raise ValueError(f"Resume setting changed: {key}")
        current = dict(wrapper.sam_audio.named_parameters())
        saved = checkpoint["trainable_model"]
        if set(saved) != {n for n, p in current.items() if p.requires_grad}:
            raise ValueError("Trainable parameter names differ from resume checkpoint")
        with torch.no_grad():
            for name, value in saved.items():
                current[name].copy_(value.to(current[name]))
        saved_aux = checkpoint.get("referring_aux")
        if (saved_aux is None) != (wrapper.referring_aux is None):
            raise ValueError("Resume auxiliary architecture differs")
        if wrapper.referring_aux is not None:
            wrapper.referring_aux.load_state_dict(saved_aux, strict=True)
        if (checkpoint.get("ema") is None) != (wrapper.ema is None):
            raise ValueError("Resume EMA configuration differs")
        if wrapper.ema is not None:
            wrapper.ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, start_epoch, skip_updates = (
            checkpoint["global_step"],
            checkpoint["epoch"],
            checkpoint["next_update"],
        )
        rng = checkpoint["rng_by_rank"][rank]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    network = wrapper
    if config.get("compile", False):
        network = torch.compile(
            network, dynamic=bool(config.get("compile_dynamic", True))
        )
    if world > 1:
        network = DDP(
            network,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    wrapper.train()
    if not resume:
        torch.manual_seed(config["seed"] + rank)

    def emit(value):
        if rank == 0:
            print(json.dumps(value), flush=True)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(value) + "\n")

    def barrier():
        if world > 1:
            dist.barrier()
        timer.sync()

    if has_validation and config.get("validate_before_training", True):
        emit(
            {
                "event": "validation",
                "step": step,
                **validate(wrapper, catalog, store, config, device, dtype, rank, world),
            }
        )
    probe_before = training_probe(
        wrapper, catalog, store, config, device, dtype, rank, world
    )
    if probe_before is not None:
        emit({"event": "training_probe_before", "step": step, **probe_before})
    # Isolate warmup, validation, checkpoints and diagnostic synchronization.
    warmup_end = step + int(config.get("benchmark_warmup_updates", 5))
    windows, local_targets, local_audio = [], 0, 0.0
    all_local_targets, all_local_slots, all_local_audio = 0, 0, 0.0
    worker_seconds = 0.0
    loss_sum = torch.zeros((), device=device)
    component_sums = torch.zeros(2, device=device)

    def flush_window():
        nonlocal local_targets, local_audio, worker_seconds, loss_sum
        timer.sync()
        elapsed = time.perf_counter() - window_start
        counts = torch.tensor(
            [local_targets, local_audio, worker_seconds],
            dtype=torch.float64,
            device=device,
        )
        duration = torch.tensor(elapsed, dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(counts)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            dist.all_reduce(loss_sum)
            dist.all_reduce(component_sums)
        record = {
            "seconds": duration.item(),
            "targets": int(counts[0].item()),
            "audio_seconds": counts[1].item(),
            "worker_read_cpu_seconds_sum": counts[2].item(),
            "loss": (loss_sum / counts[0].clamp_min(1)).item(),
            "fm_loss": (component_sums[0] / counts[0].clamp_min(1)).item(),
            "weighted_referring_loss": (component_sums[1] / counts[0].clamp_min(1)).item(),
            "step": step,
        }
        if record["targets"] and not diagnostic:
            windows.append(record)
        if record["targets"]:
            emit(
                {
                    "event": "diagnostic_window" if diagnostic else "throughput_window",
                    **record,
                }
            )
        local_targets, local_audio, worker_seconds = 0, 0.0, 0.0
        loss_sum.zero_()
        component_sums.zero_()

    profile = nullcontext()
    if diagnostic and config.get("torch_profiler", True):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=int(config.get("profile_warmup_updates", 5)),
                warmup=1,
                active=3,
                repeat=1,
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(
                str(output / f"trace-rank{rank}.json")
            ),
            record_shapes=True,
            profile_memory=True,
        )
    hook_stack = []

    def before_dit(module, inputs):
        ctx = timer.measure("dit_forward_inclusive")
        ctx.__enter__()
        hook_stack.append(ctx)

    def after_dit(module, inputs, result):
        hook_stack.pop().__exit__(None, None, None)

    hooks = []
    if diagnostic:
        hooks = [
            wrapper.sam_audio.transformer.register_forward_pre_hook(before_dit),
            wrapper.sam_audio.transformer.register_forward_hook(after_dit),
        ]
    optimizer.zero_grad(set_to_none=True)
    barrier()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    window_start = training_start = time.perf_counter()
    next_epoch, next_update = start_epoch, skip_updates
    try:
        with profile as profiler:
            for epoch in range(start_epoch, epochs):
                plan.set_epoch(epoch, skip_updates if epoch == start_epoch else 0)
                iterator = iter(loader)
                while step < max_steps:
                    timer.enabled = diagnostic and step >= int(
                        config.get("profile_warmup_updates", 5)
                    )
                    try:
                        with timer.measure("data_wait_main_process"):
                            work = next(iterator)
                    except StopIteration:
                        break
                    with timer.measure("features_prepare_inclusive"):
                        if work["cached"]:
                            with timer.measure("cached_features_transfer_to_gpu"):
                                features = move_features(work["features"], device)
                        else:
                            features = collate_features(
                                [extractor.sample(s) for s in work["samples"]]
                            )
                        valid = work["valid"].to(device, non_blocking=True)
                    syncing = work["last_micro"]
                    sync_context = (
                        network.no_sync()
                        if world > 1 and not syncing
                        else nullcontext()
                    )
                    with sync_context:
                        with torch.autocast(
                            device.type, dtype=dtype, enabled=device.type == "cuda"
                        ):
                            with timer.measure("condition_and_dit_forward_inclusive"):
                                losses = network(features)
                            with timer.measure("loss_reduction"):
                                loss = ddp_weighted_loss(
                                    losses,
                                    valid,
                                    global_valid=work["denominator"],
                                    world_size=world,
                                )
                        with timer.measure("backward_including_ddp_communication"):
                            loss.backward()
                    real_count = int(work["valid"].sum().item())
                    audio_count = float((work["duration"] * work["valid"]).sum().item())
                    all_local_targets += real_count
                    all_local_slots += work["valid"].numel()
                    all_local_audio += audio_count
                    if step >= warmup_end:
                        local_targets += real_count
                        local_audio += audio_count
                        worker_seconds += work["worker_read_seconds"]
                        loss_sum += (losses.detach() * valid).sum()
                        if wrapper.last_loss_terms is not None:
                            component_sums += torch.stack([(v * valid).sum() for v in wrapper.last_loss_terms])
                    del features, valid, loss, losses
                    if not syncing:
                        continue
                    with timer.measure("clip_grad_and_optimizer"):
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            float(config["max_grad_norm"]),
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        if wrapper.ema is not None:
                            wrapper.ema.update(wrapper)
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    step += 1
                    next_epoch, next_update = epoch, work["update"] + 1
                    if next_update == plan.updates_per_epoch:
                        next_epoch, next_update = epoch + 1, 0
                    if diagnostic and profiler is not None:
                        profiler.step()
                    validate_now = (
                        has_validation
                        and bool(config.get("validate_every", 0))
                        and step % config["validate_every"] == 0
                    )
                    save_epoch = (
                        bool(config.get("save_at_epoch_end", False))
                        and next_update == 0
                        and not diagnostic
                    )
                    save_now = save_epoch or (
                        bool(config.get("save_every", 0))
                        and step % config["save_every"] == 0
                        and not diagnostic
                    )
                    report_now = step % int(config.get("log_every", 20)) == 0
                    if (
                        step == warmup_end
                        or report_now
                        or validate_now
                        or save_now
                        or step == max_steps
                        or next_update == 0
                    ):
                        if step > warmup_end:
                            flush_window()
                        if save_now:
                            save_checkpoint(
                                output / (f"epoch-{next_epoch}.pt" if save_epoch
                                          else f"step-{step}.pt"),
                                wrapper,
                                optimizer,
                                scheduler,
                                config,
                                tuning,
                                step=step,
                                next_epoch=next_epoch,
                                next_update=next_update,
                                world=world,
                                device=device,
                                data_identity=identity,
                            )
                            if rank == 0:
                                prune_step_checkpoints(
                                    output, int(config.get("keep_last_step_checkpoints", 0))
                                )
                        if validate_now:
                            emit(
                                {
                                    "event": "validation",
                                    "step": step,
                                    **validate(
                                        wrapper,
                                        catalog,
                                        store,
                                        config,
                                        device,
                                        dtype,
                                        rank,
                                        world,
                                    ),
                                }
                            )
                        barrier()
                        window_start = time.perf_counter()
                if step >= max_steps:
                    break
    finally:
        for hook in hooks:
            hook.remove()
    timer.sync()
    inclusive_seconds = time.perf_counter() - training_start
    # Persist the last successful update before optional evaluation/reporting:
    # an evaluation OOM must not discard the completed training state.
    if not diagnostic:
        save_checkpoint(
            output / "final.pt", wrapper, optimizer, scheduler, config, tuning,
            step=step, next_epoch=next_epoch, next_update=next_update,
            world=world, device=device, data_identity=identity,
        )
    if has_validation:
        emit(
            {
                "event": "validation_final",
                "step": step,
                **validate(wrapper, catalog, store, config, device, dtype, rank, world),
            }
        )
    probe_final = training_probe(
        wrapper, catalog, store, config, device, dtype, rank, world
    )
    if probe_final is not None:
        emit({"event": "training_probe_final", "step": step, **probe_final})
    atomic_json(
        output / f"profile-rank{rank}.json",
        {
            "rank": rank,
            "stages": timer.report(),
            "memory": device_memory(device),
            "note": "Stage times are synchronized and nested/inclusive. Backward includes overlapping DDP communication; inspect NCCL kernels in trace. Not steady-state throughput.",
        },
    )
    memories = gather_objects({"rank": rank, **device_memory(device)}, world)
    exposure = torch.tensor(
        [all_local_targets, all_local_slots, all_local_audio],
        device=device,
        dtype=torch.float64,
    )
    if world > 1:
        dist.all_reduce(exposure)
    real, slots, audio_seconds = exposure.cpu().tolist()
    if rank == 0:
        atomic_json(
            output / "summary.json",
            {
                "mode": mode,
                "features": feature_mode,
                "steps": step,
                "train_targets": len(dataset),
                "validation_targets": validation_targets,
                "validation_enabled": has_validation,
                "processed_target_examples_including_warmup": int(real),
                "processed_slots_including_dummy": int(slots),
                "dummy_fraction": (slots - real) / slots if slots else 0.0,
                "processed_audio_seconds_including_warmup": audio_seconds,
                "training_loop_inclusive_seconds": inclusive_seconds,
                "throughput": None
                if diagnostic
                else throughput_summary(windows, world),
                "windows": windows,
                "memory_by_rank": memories,
                "global_batch_nominal": world
                * config["batch_size"]
                * config["gradient_accumulation_steps"],
                "timing_note": "Throughput excludes warmup, validation, saves and reporting collectives; loop time includes them. Cache preparation is a separate recorded command.",
            },
        )
def main():
    main_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("prepare", "profile", "train")
    )
    parser.add_argument("--features", choices=("online", "cached"), default="cached")
    parser.add_argument(
        "--run-name",
        required=True,
        help="New local output subdirectory; never overwritten",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run-name must be one directory name")
    if config.get("dtype", "bfloat16") not in ("bfloat16", "float32"):
        raise ValueError("This runner supports BF16/FP32, not unscaled FP16")
    config.setdefault("dtype", "bfloat16")
    if config.get("fixed_train_rng_seed") is not None:
        raise ValueError(
            "Fixed training noise is a smoke-test trick, not supported here"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This CLI requires a CUDA GPU; CPU correctness tests are separate"
        )
    from huggingface_hub import snapshot_download
    from sam_audio import SAMAudio, SAMAudioProcessor
    from .model import configure_native_tuning, native_tuning_config_from_mapping
    from .train import _make_dataset, _resolve_dataset

    rank, world, local = [
        int(os.environ.get(k, d))
        for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    ]
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    output_created = completed = False
    try:
        output = Path(config["output_dir"]) / args.run_name
        # Every rank checks before rank 0 creates it, then synchronize the outcome.
        exists = output.exists() if rank == 0 else None
        if world > 1:
            status = [exists]
            dist.broadcast_object_list(status, src=0)
            exists = status[0]
        if exists:
            raise FileExistsError(f"Choose a new run-name; preserving {output}")
        output.mkdir(parents=True, exist_ok=True)
        output_created = True
        began = time.perf_counter()
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(int(config.get("cpu_threads_per_rank", 4)))
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
        dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
        model_path = config.get("model_local_path")
        if not model_path:
            # Resolve once so the model and processor use the same snapshot.
            model_path = snapshot_download(
                config["sam_audio_model"],
                revision=config.get("model_revision", "main"),
                allow_patterns=["checkpoint.pt", "config.json"],
                token=os.environ.get("HF_TOKEN"),
            )
        sam = SAMAudio.from_pretrained(
            model_path,
            visual_ranker=None,
            text_ranker=None,
            span_predictor=None,
            strict=True,
        )
        if config.get("target_only", False):
            from .target_only import convert_target_only
            convert_target_only(sam)
        processor = SAMAudioProcessor.from_pretrained(model_path)
        sam.to(device=device, dtype=dtype).requires_grad_(False).eval()
        signature = model_signature(model_path, config, processor, sam)
        atomic_json(output / f"pretrained-load-rank{rank}.json", sam._pretrained_load_report)
        store = ComponentStore(
            config["feature_cache_dir"],
            signature,
            lru_entries=int(config.get("cache_lru_entries_per_worker", 8)),
        )
        root, manifest = _resolve_dataset(config)
        datasets = {
            split: _make_dataset(
                config,
                split=split,
                data_root=root,
                manifest=manifest,
                sample_rate=processor.audio_sampling_rate,
            )
            for split in training_splits(config)
        }
        protected_eval = config.get("protected_eval_manifest")
        if args.mode == "train" and not protected_eval:
            raise ValueError("Training requires protected_eval_manifest to reject test clips/sources before optimization")
        if protected_eval:
            eval_path = Path(protected_eval)
            if not eval_path.is_absolute():
                eval_path = root / eval_path
            atomic_json(output / f"data-separation-rank{rank}.json",
                        audit_training_separation(datasets["train"], eval_path,
                            temporal_disjoint_datasets=config.get("temporal_disjoint_eval_datasets", [])))
        plan = make_cache_plan(datasets)
        timer = StageTimer(device, enabled=args.mode == "prepare")
        extractor = FrozenExtractor(
            sam,
            processor,
            device,
            dtype,
            timer,
            to_cpu=args.mode == "prepare",
            video_frame_chunk=int(config.get("video_frame_chunk", 0)),
        )
        atomic_json(
            output / f"setup-rank{rank}.json",
            {
                "seconds": time.perf_counter() - began,
                "device": torch.cuda.get_device_name(device),
                "signature": signature,
                "resolved_model_path": str(model_path),
                "plan_id": plan["id"],
            },
        )
        if rank == 0:
            atomic_json(output / "config.json", config)
        if args.mode == "prepare":
            result = build_components(
                plan, store, extractor, rank=rank, world_size=world
            )
            atomic_json(output / f"cache-prepare-rank{rank}.json", result)
            if world > 1:
                dist.barrier()
            if rank == 0:
                path = finalize_catalog(plan, store, processor.audio_sampling_rate)
                atomic_json(
                    output / "cache-catalog.json",
                    {
                        "path": str(path),
                        "namespace": str(store.root),
                        "mixtures": len(plan["mixtures"]),
                        "videos": sum(
                            len(m["videos"]) for m in plan["mixtures"].values()
                        ),
                        "targets": sum(
                            len(m["targets"]) for m in plan["mixtures"].values()
                        ),
                        "prompts": len(plan["texts"]),
                    },
                )
            completed = True
            return
        catalog_path = store.root / f"catalog-{plan['id']}.json"
        if not catalog_path.exists():
            raise FileNotFoundError(
                "Matching feature cache missing; run --mode prepare first: "
                + str(catalog_path)
            )
        catalog = json.loads(catalog_path.read_text())
        if catalog["signature"] != signature:
            raise ValueError("Cache signature mismatch")
        tuning = configure_native_tuning(sam, native_tuning_config_from_mapping(config))
        wrapper = CachedFlowModel(sam, float(config["sigma_min"]))
        # Optional feature/loss parity before cached training. Checks gradients
        # for the first example by default, without updating parameters. A zero
        # target count intentionally skips this expensive online-video check.
        parity = []
        parity_started = time.perf_counter()
        for index in range(
            min(int(config.get("cache_parity_targets", 1)), len(datasets["train"]))
        ):
            ds = datasets["train"]
            ds.set_epoch(0)
            sample = ds[index]
            with torch.no_grad(), extractor.autocast():
                condition = processor(
                    descriptions=[sample["prompt"]],
                    audios=[sample["mixture"]],
                    masked_videos=[sample["video_path"]],
                ).to(device)
                forward_args = sam._get_forward_args(
                    condition, video_frame_chunk=extractor.video_frame_chunk
                )
                clean = extractor.clean(sample["target"], sample.get("residual"))
            del (
                condition
            )  # Free full-resolution GPU video frames before gradient parity.
            refs = catalog["splits"]["train"]["entries"][index]["variants"]
            from .optimized_utils import select_variant

            cached = store.assemble(refs[select_variant(ds.seed, 0, index, len(refs))])
            with extractor.autocast():
                parity.append(
                    verify_cached_example(
                        wrapper,
                        {"forward_args": forward_args, "clean": clean},
                        cached,
                        device=device,
                        check_gradients=index == 0
                        and bool(config.get("cache_parity_gradients", True)),
                    )
                )
            del forward_args, clean, cached, sample
        atomic_json(
            output / f"parity-rank{rank}.json",
            {
                "seconds": time.perf_counter() - parity_started,
                "examples": parity,
                "skipped": not parity,
                "reference_video_frame_chunk": extractor.video_frame_chunk,
                "reference": "independent official processor/forward with declared cache chunk size",
            },
        )
        if rank == 0:
            atomic_json(output / "tuning_summary.json", tuning)
        # Reset RNG after checks so online/cached runs start identically.
        torch.manual_seed(config["seed"] + rank)
        run_training(
            config,
            mode=args.mode,
            feature_mode=args.features,
            wrapper=wrapper,
            extractor=extractor,
            catalog=catalog,
            store=store,
            native_dataset=datasets["train"],
            tuning=tuning,
            output=output,
            device=device,
            dtype=dtype,
            rank=rank,
            world=world,
        )
        completed = True
    finally:
        if output_created:
            atomic_json(
                output / f"lifecycle-rank{rank}.json",
                {
                    "mode": args.mode,
                    "completed": completed,
                    "command_main_seconds": time.perf_counter() - main_started,
                    "note": "Includes setup/downloads, parity, training/cache work, validation and final saves; excludes Python import/remote container bootstrap and process-group teardown.",
                },
            )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
