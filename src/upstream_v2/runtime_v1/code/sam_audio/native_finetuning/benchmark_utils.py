"""Small, testable batching primitives for the native DDP throughput benchmark."""

from __future__ import annotations

import random

import torch
from torch.nn.utils.rnn import pad_sequence


def global_batches(count, *, world_size, micro_batch, epochs, seed, ordered=False):
    """Yield (epoch, index-or-None) slots; None contributes zero loss/count.

    Unlike DistributedSampler's padding, no duplicated target is counted as an
    additional training example. All ranks still participate in every update.
    """
    if min(count, world_size, micro_batch, epochs) < 1:
        raise ValueError("Batch dimensions and epoch count must be positive")
    width = world_size * micro_batch
    for epoch in range(epochs):
        indices = list(range(count))
        if not ordered:
            random.Random(seed + epoch).shuffle(indices)
        for start in range(0, count, width):
            batch = indices[start : start + width]
            yield [(epoch, i) for i in batch] + [(epoch, None)] * (width - len(batch))


def ddp_weighted_loss(per_example_loss, valid, *, global_valid, world_size):
    """DDP averages rank gradients, so compensate to get a global target mean."""
    if global_valid < 1:
        raise ValueError("An optimizer update must contain a real example")
    return (per_example_loss * valid).sum() * (world_size / global_valid)


def per_target_mse(prediction, velocity, mask):
    if prediction.ndim != 3 or prediction.shape != velocity.shape:
        raise ValueError("Prediction and velocity must have identical [B, T, C] shapes")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("Audio mask must match prediction [B, T]")
    # Preserve batch=1 + gradient-accumulation semantics: each target has equal
    # weight, rather than letting long clips dominate when micro_batch > 1.
    error = (prediction.float() - velocity.float()).square()
    mask = mask.bool()
    denominator = mask.sum(1).clamp_min(1) * prediction.shape[-1]
    return (error * mask.unsqueeze(-1)).sum((1, 2)) / denominator


def collate_features(items):
    """Batch individually encoded features, preserving codec boundary effects.

    Audio/video/clean latents share T; text has its own length L. SAM Audio uses
    True for valid mask positions and anchor slot 1 for the padding anchor.
    """
    for item in items:
        audio, clean = item["forward_args"]["audio_features"], item["clean"]
        if audio.shape[:2] != clean.shape[:2] or (
            audio.shape[-1] != clean.shape[-1]
            and (audio.shape[-1], clean.shape[-1]) != (256, 128)
        ):
            raise ValueError("Cached mixture/target feature shapes disagree")
    args = {}
    keys = items[0]["forward_args"].keys()
    for key in keys:
        values = [item["forward_args"][key].squeeze(0) for item in items]
        if key == "masked_video_features":
            args[key] = pad_sequence(
                [v.transpose(0, 1) for v in values], batch_first=True
            ).transpose(1, 2)
        elif key == "anchor_ids":
            args[key] = pad_sequence(values, batch_first=True, padding_value=3)
        else:
            args[key] = pad_sequence(
                values, batch_first=True,
                padding_value=1 if key == "anchor_alignment" else 0,
            )
    clean = pad_sequence([i["clean"].squeeze(0) for i in items], batch_first=True)
    if args["audio_features"].shape[:2] != clean.shape[:2]:
        raise ValueError("Cached mixture/target feature shapes disagree")
    if args["masked_video_features"].shape[-1] != clean.shape[1]:
        raise ValueError("Cached video is not time-aligned")
    result = {"forward_args": args, "clean": clean}
    if any("referring_negatives" in i for i in items):
        if not all("referring_negatives" in i for i in items):
            raise ValueError("Mixed referring/non-referring batch")
        from .referring_aux import collate_negatives
        result["referring_negatives"], result["referring_negative_valid"] = collate_negatives(items, clean.shape[1])
    return result
