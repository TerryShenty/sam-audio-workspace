"""Dependency-light helpers for cached native training (also tested on CPU).

Engineering references, not a replacement for SAM Audio's loss/model:
StableTTS/datas/sampler.py: distributed length buckets.
MMAudio/mmaudio/runner.py: cached conditions and optional compiled training.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


class StageTimer:
    """Diagnostic wall times with CUDA synchronization, NOT throughput timings.

    Enabled only in the separate profile command/cache preparation. Nested
    stages are inclusive and must not be added together. Throughput mode does
    not synchronize here; it times windows bounded by CUDA synchronization.
    record_function ranges also expose forward/backward/NCCL in profiler traces.
    """

    def __init__(self, device="cpu", enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.samples = defaultdict(list)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        with torch.profiler.record_function(name):
            self.sync()
            start = time.perf_counter()
            try:
                yield
            finally:
                self.sync()
                self.samples[name].append(time.perf_counter() - start)

    def report(self):
        return {
            name: {
                "calls": len(values),
                "total_seconds": sum(values),
                "mean_seconds": sum(values) / len(values),
                "max_seconds": max(values),
            }
            for name, values in sorted(self.samples.items())
        }


class LengthBucketPlan:
    """One target/epoch, no omitted or overweight duplicate samples.

    Every rank independently constructs the same global microbatches. A slot is
    (epoch, index, valid, update_denominator, last_microbatch, update_index).
    Dummy slots run a real input but contribute zero loss. Epoch/index travel
    to workers explicitly so persistent workers sample the right paraphrase.
    Accumulation is normalized by REAL examples in the whole optimizer update.
    """

    def __init__(
        self,
        lengths,
        *,
        world_size=1,
        rank=0,
        micro_batch=1,
        accumulation=1,
        seed=0,
        bucket_batches=32,
        shuffle=True,
        exact_lengths=False,
    ):
        if not lengths or min(world_size, micro_batch, accumulation) < 1:
            raise ValueError("Nonempty data and positive batch dimensions required")
        if not 0 <= rank < world_size or bucket_batches < 0:
            raise ValueError("Invalid rank or bucket size")
        self.lengths = list(lengths)
        self.world_size, self.rank = world_size, rank
        self.micro_batch, self.accumulation = micro_batch, accumulation
        self.seed, self.bucket_batches, self.shuffle = seed, bucket_batches, shuffle
        self.exact_lengths = exact_lengths
        local_count = (
            sum(math.ceil(n / micro_batch) for n in Counter(lengths).values())
            if exact_lengths
            else math.ceil(len(lengths) / micro_batch)
        )
        self._microbatches_per_epoch = math.ceil(local_count / world_size)
        self.epoch, self.skip_updates = 0, 0

    def set_epoch(self, epoch, skip_updates=0):
        self.epoch, self.skip_updates = epoch, skip_updates

    @property
    def updates_per_epoch(self):
        return math.ceil(self._microbatches_per_epoch / self.accumulation)

    def _batches(self):
        width = self.world_size * self.micro_batch
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        if self.exact_lengths:
            # SAM Audio x_embedder has temporal GroupNorm BEFORE attention
            # masks. Right-padding different T values changes valid features.
            # Group *within each rank* by exact T, not across all GPUs: ranks
            # may have different T, without serializing every unique duration.
            groups = defaultdict(list)
            for index in indices:
                groups[self.lengths[index]].append(index)
            local_batches = []
            for group in groups.values():
                for start in range(0, len(group), self.micro_batch):
                    part = group[start : start + self.micro_batch]
                    local_batches.append(
                        [(i, True) for i in part]
                        + [(part[-1], False)] * (self.micro_batch - len(part))
                    )
            if self.shuffle:
                rng.shuffle(local_batches)
            if self.bucket_batches:
                window = self.world_size * self.bucket_batches
                local_batches = [
                    b
                    for start in range(0, len(local_batches), window)
                    for b in sorted(
                        local_batches[start : start + window],
                        key=lambda b: self.lengths[b[0][0]],
                    )
                ]
            batches = []
            for start in range(0, len(local_batches), self.world_size):
                group = local_batches[start : start + self.world_size]
                group += [[(i, False) for i, _ in group[-1]]] * (
                    self.world_size - len(group)
                )
                batches.append([slot for batch in group for slot in batch])
            if self.shuffle and self.bucket_batches:
                rng.shuffle(batches)
            return batches
        if self.bucket_batches:
            window = max(width, width * self.bucket_batches)
            indices = [
                i
                for start in range(0, len(indices), window)
                for i in sorted(
                    indices[start : start + window], key=self.lengths.__getitem__
                )
            ]
        batches = []
        for start in range(0, len(indices), width):
            part = indices[start : start + width]
            batches.append(
                [(i, True) for i in part] + [(part[-1], False)] * (width - len(part))
            )
        if self.shuffle and self.bucket_batches:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches()
        for update, start in enumerate(range(0, len(batches), self.accumulation)):
            if update < self.skip_updates:
                continue
            group = batches[start : start + self.accumulation]
            denominator = sum(valid for b in group for _, valid in b)
            for micro, batch in enumerate(group):
                local = batch[
                    self.rank * self.micro_batch : (self.rank + 1) * self.micro_batch
                ]
                yield [
                    (self.epoch, i, valid, denominator, micro == len(group) - 1, update)
                    for i, valid in local
                ]

    def __len__(self):
        total = self._microbatches_per_epoch
        return max(0, total - self.skip_updates * self.accumulation)


def select_variant(seed, epoch, index, count):
    return random.Random(seed + epoch * 1_000_003 + index).randrange(count)


def move_features(batch, device):
    result = {
        "clean": batch["clean"].to(device, non_blocking=True),
        "forward_args": {
            k: v.to(device, non_blocking=True) for k, v in batch["forward_args"].items()
        },
    }

    for key in ("referring_negatives", "referring_negative_valid"):
        if key in batch:
            result[key] = batch[key].to(device, non_blocking=True)
    return result

def device_memory(device):
    if torch.device(device).type != "cuda":
        return {}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_total_gib": torch.cuda.get_device_properties(device).total_memory
        / 2**30,
    }


def throughput_summary(windows, world_size):
    """Window time is the slowest rank; example count is summed across ranks."""
    seconds = sum(w["seconds"] for w in windows)
    targets = sum(w["targets"] for w in windows)
    audio_seconds = sum(w["audio_seconds"] for w in windows)
    return {
        "world_size": world_size,
        "timed_targets": targets,
        "timed_seconds": seconds,
        "targets_per_second": targets / seconds if seconds else None,
        "seconds_per_target": seconds / targets if targets else None,
        "timed_audio_seconds": audio_seconds,
        "audio_seconds_per_second": audio_seconds / seconds if seconds else None,
    }
