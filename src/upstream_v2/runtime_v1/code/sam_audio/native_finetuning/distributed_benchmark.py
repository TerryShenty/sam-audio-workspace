"""Bounded, real-media DDP benchmark; not a new official SAM Audio recipe.

Run with torchrun. The launcher runs each phase in a fresh process group, so
an optional batch-size/OOM probe cannot corrupt the successful primary runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from sam_audio import SAMAudio, SAMAudioProcessor
from .benchmark_utils import collate_features, ddp_weighted_loss, global_batches, per_target_mse
from .data import collate_native_referring
from .model import NativeSAMAudioFineTuner, configure_native_tuning, native_tuning_config_from_mapping, trainable_parameter_groups
from .objective import make_flow_matching_state
from .train import _make_condition_batch, _make_dataset, _stable_batch_seed, _write_json


def selected_row(dataset, epoch, index):
    rows = dataset.examples[index]
    rng = random.Random(dataset.seed + epoch * 1_000_003 + index)
    return rows[rng.randrange(len(rows))]


def cache_key(dataset, epoch, index):
    row = selected_row(dataset, epoch, index)
    # Cache lives in a unique job directory. Full row includes prompt, paths,
    # target identity; preprocessing/model identity are in cache_manifest.json.
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def load_feature(cache_dir, dataset, epoch, index):
    path = cache_dir / (cache_key(dataset, epoch, index) + ".pt")
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def encode_sample(wrapper, processor, sample, device):
    batch = collate_native_referring([sample])
    condition = _make_condition_batch(processor, batch, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        args = wrapper.sam_audio._get_forward_args(condition)
        clean = wrapper._encode_joint_target(batch["target"].to(device), batch["residual"].to(device))
    if clean.shape != args["audio_features"].shape:
        raise ValueError("Encoded target/mixture shapes disagree")
    return {"forward_args": {k: v.detach().cpu() for k, v in args.items()},
            "clean": clean.detach().cpu(), "sample_id": sample["sample_id"],
            "prompt": sample["prompt"], "duration": sample["mixture"].shape[-1] / processor.audio_sampling_rate}


class WorkDataset(Dataset):
    def __init__(self, base, cache_dir=None, index_order=None):
        self.base, self.cache_dir, self.index_order = base, cache_dir, index_order

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        epoch, index, global_valid = item
        valid = index is not None
        index = index if valid else 0
        if self.index_order is not None:
            index = self.index_order[index]
        if self.cache_dir is None:
            self.base.set_epoch(epoch)
            value = self.base[index]
        else:
            value = load_feature(self.cache_dir, self.base, epoch, index)
        return value, float(valid), global_valid


def collate_work(items):
    values, valid, counts = zip(*items)
    assert len(set(counts)) == 1
    cached = "clean" in values[0]
    batch = collate_features(values) if cached else collate_native_referring(list(values))
    durations = [v["duration"] if cached else v["mixture"].shape[-1] / 48000 for v in values]
    return {"batch": batch, "valid": torch.tensor(valid), "global_valid": counts[0],
            "durations": torch.tensor(durations), "cached": cached}


class LossWrapper(NativeSAMAudioFineTuner):
    def __init__(self, sam_audio, processor, *, sigma_min):
        super().__init__(sam_audio, sigma_min=sigma_min)
        self.processor = processor

    def forward(self, work):
        device = next(self.sam_audio.parameters()).device
        batch = work["batch"]
        if work["cached"]:
            forward_args = {k: v.to(device, non_blocking=True) for k, v in batch["forward_args"].items()}
            clean = batch["clean"].to(device, non_blocking=True)
        else:
            condition = _make_condition_batch(self.processor, batch, device)
            with torch.no_grad():
                forward_args = self.sam_audio._get_forward_args(condition)
                clean = self._encode_joint_target(batch["target"].to(device), batch["residual"].to(device))
        state = make_flow_matching_state(clean, sigma_min=self.sigma_min)
        prediction = self.sam_audio(noisy_audio=state.noisy_latent, time=state.time, **forward_args)
        losses = per_target_mse(prediction, state.target_velocity, forward_args["audio_pad_mask"])
        return losses, prediction.detach().float().square().mean().sqrt(), state.target_velocity.detach().float().square().mean().sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=["online", "cache", "cached", "no_checkpoint", "batch2"], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    started = time.perf_counter()
    output = Path(config["output_dir"]) / args.phase
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["feature_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(int(config.get("cpu_threads_per_rank", 4)))

    def emit(record):
        record = {"phase": args.phase, **record}
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")

    emit({"event": "phase_start", "world_size": world})
    model_source = config.get("model_local_path", config["sam_audio_model"])
    sam = SAMAudio.from_pretrained(model_source, token=os.environ.get("HF_TOKEN"),
                                  visual_ranker=None, text_ranker=None, span_predictor=None, strict=True)
    processor = SAMAudioProcessor.from_pretrained(model_source)
    sam.to(device=device, dtype=torch.bfloat16)
    checkpointing = args.phase != "no_checkpoint"
    tuning = configure_native_tuning(sam, native_tuning_config_from_mapping({**config, "gradient_checkpointing": checkpointing}))
    # Explicit assignment also covers an already-constructed transformer.
    sam.transformer.gradient_checkpointing = checkpointing
    wrapper = LossWrapper(sam, processor, sigma_min=config["sigma_min"])
    wrapper.train()
    train = _make_dataset(config, split="train", data_root=Path(config["data_root"]),
                          manifest=Path(config["data_root"]) / config["manifest"], sample_rate=processor.audio_sampling_rate)
    val = _make_dataset(config, split="validation", data_root=Path(config["data_root"]),
                        manifest=Path(config["data_root"]) / config["manifest"], sample_rate=processor.audio_sampling_rate)
    if rank == 0:
        _write_json(output / "tuning_summary.json", tuning)
        _write_json(output / "data_summary.json", {"train_targets": len(train), "validation_targets": len(val),
                    "train_sample_ids": [f"{x[0].get('dataset', '')}::{x[0]['clip_id']}::{x[0]['target_source_id']}" for x in train.examples]})
    if args.phase == "cache":
        wrapper.eval()
        jobs = {}
        for ds in (train, val):
            for epoch in range(4 if ds is train else 1):
                for index in range(len(ds)):
                    jobs.setdefault(cache_key(ds, epoch, index), (ds, epoch, index))
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for n, (key, (ds, epoch, index)) in enumerate(sorted(jobs.items())):
            if n % world != rank:
                continue
            ds.set_epoch(epoch)
            features = encode_sample(wrapper, processor, ds[index], device)
            temporary = cache_dir / f"{key}.rank{rank}.tmp"
            torch.save(features, temporary)
            temporary.replace(cache_dir / f"{key}.pt")
            if n % (5 * world) == 0:
                emit({"event": "cache_progress", "assigned_global_position": n, "total_unique": len(jobs)})
        dist.barrier()
        torch.cuda.synchronize()
        cache_seconds = time.perf_counter() - begin
        if rank == 0:
            missing = [k for k in jobs if not (cache_dir / f"{k}.pt").exists()]
            if missing:
                raise RuntimeError(f"Incomplete cache: {len(missing)} missing entries")
            summary = {"status": "completed", "unique_sample_prompt_features": len(jobs), "cache_preparation_seconds": cache_seconds,
                       "cache_bytes": sum(p.stat().st_size for p in cache_dir.glob("*.pt")),
                       "model": config["sam_audio_model"], "max_seconds": config["max_seconds"],
                       "dataset_revision": config["benchmark_dataset_revision"], "dtype": "bfloat16",
                       "phase_wall_seconds": time.perf_counter() - started}
            _write_json(output / "summary.json", summary)
            emit({"event": "summary", **summary})
        dist.destroy_process_group()
        return

    cached = args.phase != "online"
    if cached:
        # Compare reloaded cache with an actual fresh frozen-encoder forward on
        # every rank before relying on it. No text/projection/DiT feature after
        # a trainable layer is cached.
        wrapper.eval()
        train.set_epoch(0)
        index = rank % len(train)
        actual = encode_sample(wrapper, processor, train[index], device)
        saved = load_feature(cache_dir, train, 0, index)
        errors = {}
        for key in actual["forward_args"]:
            a, b = actual["forward_args"][key], saved["forward_args"][key]
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
            errors[key] = float((a.float() - b.float()).abs().max())
        torch.testing.assert_close(actual["clean"], saved["clean"], rtol=1e-3, atol=1e-3)
        _write_json(output / f"cache_parity_rank{rank}.json", errors)
        del actual, saved
        wrapper.train()

    ddp = DDP(wrapper, device_ids=[local_rank], output_device=local_rank,
              broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    groups = trainable_parameter_groups(sam, lora_learning_rate=config["lora_learning_rate"],
              condition_learning_rate=config["condition_learning_rate"], block_learning_rate=config["block_learning_rate"],
              weight_decay=config["weight_decay"])
    optimizer = torch.optim.AdamW(groups, betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., max(step, 1) / config["warmup_steps"]))
    micro = 2 if args.phase == "batch2" else 1
    full_run = args.phase in ("online", "cached", "batch2")
    max_steps = 4 * ((len(train) + world * micro - 1) // (world * micro)) if full_run else 10
    plans = list(global_batches(len(train), world_size=world, micro_batch=micro, epochs=4,
                               seed=seed))[:max_steps]
    local_batches = []
    for plan in plans:
        valid_count = sum(i is not None for _, i in plan)
        local_batches.append([(e, i, valid_count) for e, i in plan[rank*micro:(rank+1)*micro]])
    dataset = WorkDataset(train, cache_dir if cached else None)
    loader = DataLoader(dataset, batch_sampler=local_batches, collate_fn=collate_work,
                        num_workers=2, pin_memory=True, persistent_workers=False,
                        multiprocessing_context="spawn")

    @torch.no_grad()
    def validate(step):
        wrapper.eval()
        stats = torch.zeros(2, device=device, dtype=torch.float64)
        for i in range(rank, len(val), world):
            val.set_epoch(0)
            value = load_feature(cache_dir, val, 0, i) if cached else val[i]
            work = collate_work([(value, 1., 1)])
            sample_id = value["sample_id"]
            with torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(_stable_batch_seed(config["validation_seed"], [sample_id]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, _ = wrapper(work)
            stats += torch.tensor([loss.item(), 1], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        wrapper.train()
        result = (stats[0] / stats[1]).item()
        emit({"event": "validation", "step": step, "flow_loss": result, "targets": int(stats[1].item())})
        return result

    torch.manual_seed(seed + rank)
    initial_val = validate(0) if full_run else None
    initial_trainable = {n: p.detach().cpu().clone() for n, p in sam.named_parameters() if p.requires_grad} if rank == 0 else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    torch.cuda.synchronize()
    emit({"event": "training_start", "micro_batch_per_gpu": micro, "gradient_accumulation": 1,
          "effective_global_batch": micro * world, "steps": len(plans), "checkpointing": checkpointing,
          "feature_cached": cached, "warmup_excluded_steps": 5, "trainable_parameters": tuning["trainable_parameters"]})
    all_seconds = 0.
    timed_seconds = 0.
    all_targets = 0
    timed_targets = 0
    timed_audio = 0.
    matched_seconds, matched_targets, matched_audio = 0., 0, 0.
    epoch_losses = {}
    training_begin = time.perf_counter()
    iterator = iter(loader)
    final_val = None
    for step in range(1, len(plans) + 1):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        work = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, pred_rms, velocity_rms = ddp(work)
            valid = work["valid"].to(device)
            loss = ddp_weighted_loss(losses, valid, global_valid=work["global_valid"], world_size=world)
        loss.backward()
        missing = [n for n, p in sam.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError(f"Trainable parameters missing gradients: {missing[:8]}")
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in sam.parameters() if p.requires_grad], config["max_grad_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite loss or gradients")
        optimizer.step()
        scheduler.step()
        stats = torch.tensor([(losses.detach()*valid).sum().item(), valid.sum().item(),
                              (work["durations"]*work["valid"]).sum().item(),
                              pred_rms.item(), velocity_rms.item()], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - begin, device=device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds, count = elapsed.item(), int(stats[1].item())
        all_seconds += seconds
        all_targets += count
        if step > 5:
            timed_seconds += seconds
            timed_targets += count
            timed_audio += stats[2].item()
        epoch = plans[step-1][0][0]
        # For B=1 versus B=2 exclude the entire first epoch, not five updates
        # (five updates represent different amounts of data at different B).
        # Both full runs then time the identical remaining 351 target examples.
        if epoch >= 1:
            matched_seconds += seconds
            matched_targets += count
            matched_audio += stats[2].item()
        sums = epoch_losses.setdefault(str(epoch), [0., 0])
        sums[0] += stats[0].item()
        sums[1] += count
        emit({"event": "train_step", "step": step, "epoch": epoch, "targets": count,
              "flow_loss": (stats[0]/stats[1]).item(), "seconds": seconds,
              "grad_norm": grad_norm.item(), "prediction_rms": (stats[3]/world).item(),
              "velocity_rms": (stats[4]/world).item()})
        if full_run and (step % 30 == 0 or step == len(plans)):
            final_val = validate(step)
        del work, losses, loss, valid
    torch.cuda.synchronize()
    loop_wall = time.perf_counter() - training_begin
    gpu = torch.cuda.get_device_properties(device)
    rank_stats = {"rank": rank, "gpu": gpu.name, "total_gib": gpu.total_memory / 2**30,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "loop_wall_seconds": loop_wall}
    _write_json(output / f"gpu_rank{rank}.json", rank_stats)
    ranks = [None] * world
    dist.all_gather_object(ranks, rank_stats)
    if rank == 0:
        changed = [n for n, p in sam.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(), initial_trainable[n])]
        if not changed:
            raise RuntimeError("Optimizer did not change any trainable parameter")
        summary = {"status": "completed", "phase": args.phase, "world_size": world,
                   "micro_batch_per_gpu": micro, "gradient_accumulation": 1, "checkpointing": checkpointing,
                   "feature_cached": cached, "train_targets": len(train), "validation_targets": len(val),
                   "optimizer_steps": len(plans), "all_effective_examples": all_targets,
                   "all_training_seconds": all_seconds, "timed_steps": len(plans)-5,
                   "timed_effective_examples": timed_targets, "timed_training_seconds": timed_seconds,
                   "timed_targets_per_second": timed_targets / timed_seconds,
                   "matched_post_epoch0_examples": matched_targets,
                   "matched_post_epoch0_seconds": matched_seconds,
                   "matched_post_epoch0_audio_seconds": matched_audio,
                   "matched_post_epoch0_targets_per_second": matched_targets / matched_seconds if matched_seconds else None,
                   "timed_audio_seconds": timed_audio, "loop_wall_seconds": max(r["loop_wall_seconds"] for r in ranks),
                   "phase_wall_seconds_before_checkpoint": time.perf_counter()-started,
                   "initial_validation_loss": initial_val, "final_validation_loss": final_val,
                   "epoch_mean_train_loss": {e: s/c for e, (s, c) in epoch_losses.items()},
                   "changed_trainable_tensor_count": len(changed), "gpus": ranks,
                   "note": "Small train-flow-loss benchmark, not held-out source-separation quality evaluation. Cache prep and setup are excluded from timed_training_seconds."}
        if full_run:
            torch.save({"format": "sam_audio_native_ddp_benchmark_v1", "config": config,
                        "trainable_model": wrapper.trainable_state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": len(plans)}, output / "last.pt")
        _write_json(output / "summary.json", summary)
        emit({"event": "summary", **summary})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
