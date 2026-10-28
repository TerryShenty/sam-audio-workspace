"""Losses used by the first referring-prompt training baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingState:
    noisy_latent: torch.Tensor
    target_velocity: torch.Tensor
    time: torch.Tensor
    noise: torch.Tensor


def sample_flow_matching_state(
    clean_latent: torch.Tensor,
    sigma_min: float = 1.0e-5,
    *,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> FlowMatchingState:
    """Construct the straight conditional-flow path used by SAM Audio.

    Args:
        clean_latent: Joint target/residual latent ``[B, T, C]``.
        sigma_min: Small terminal noise floor from the paper equation.
        time: Optional per-example times ``[B]``; sampled uniformly otherwise.
        noise: Optional Gaussian starting point with the same shape as clean_latent.
    """

    if clean_latent.ndim != 3:
        raise ValueError(
            f"clean_latent must have shape [B, T, C], got {clean_latent.shape}"
        )
    batch_size = clean_latent.size(0)
    if time is None:
        # Keep time in fp32 even when model weights/latents use bf16; otherwise
        # the uniform distribution has unnecessarily coarse time resolution.
        time = torch.rand(batch_size, device=clean_latent.device)
    if noise is None:
        noise = torch.randn_like(clean_latent)
    if time.shape != (batch_size,):
        raise ValueError(f"time must have shape [{batch_size}], got {time.shape}")

    t = time.to(clean_latent.dtype)[:, None, None]
    scale = 1.0 - (1.0 - sigma_min) * t
    noisy_latent = scale * noise + t * clean_latent
    target_velocity = clean_latent - (1.0 - sigma_min) * noise
    return FlowMatchingState(noisy_latent, target_velocity, time, noise)


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    if valid_mask.ndim != 2 or values.shape[:2] != valid_mask.shape:
        raise ValueError(
            "valid_mask must be [B, T] and match the first two value dimensions; "
            f"got values={values.shape}, mask={valid_mask.shape}"
        )
    weights = valid_mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.expand_as(values).sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def flow_matching_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked mean-squared velocity error."""

    if prediction.shape != target_velocity.shape:
        raise ValueError(
            f"prediction and target must match: {prediction.shape} vs "
            f"{target_velocity.shape}"
        )
    return _masked_mean((prediction - target_velocity).square(), valid_mask)


def cosine_alignment_loss(
    predicted_features: torch.Tensor,
    target_features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The paper's ``1 - cosine`` target-representation objective.

    Both tensors are frame-synchronous ``[B, T, F]`` representations.
    """

    if predicted_features.shape != target_features.shape:
        raise ValueError(
            "predicted_features and target_features must match; got "
            f"{predicted_features.shape} and {target_features.shape}"
        )
    distance = 1.0 - F.cosine_similarity(
        predicted_features.float(), target_features.float(), dim=-1
    )
    return _masked_mean(distance, valid_mask)


def referring_contrastive_loss(
    predicted_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Choose the prompted source over same-video wrong-source stems.

    Shapes:
        predicted_features: ``[B, T, F]`` from an intermediate DiT block.
        positive_features: ``[B, T, F]`` for the requested source.
        negative_features: ``[B, K, T, F]`` for wrong sources in the same video.

    Cosine similarities are computed per frame and averaged over valid frames.
    Cross entropy then makes candidate zero (the positive) win over all negatives.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if predicted_features.shape != positive_features.shape:
        raise ValueError(
            f"predicted/positive shapes differ: {predicted_features.shape} vs "
            f"{positive_features.shape}"
        )
    if negative_features.ndim != 4:
        raise ValueError(
            "negative_features must have shape [B, K, T, F], got "
            f"{negative_features.shape}"
        )
    batch_size, _, frames, feature_dim = negative_features.shape
    if predicted_features.shape != (batch_size, frames, feature_dim):
        raise ValueError("negative feature dimensions do not match predicted features")
    if negative_features.size(1) < 1:
        raise ValueError("at least one wrong-source feature is required")

    query = F.normalize(predicted_features.float(), dim=-1)
    positive = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_features.float(), dim=-1)

    positive_similarity = (query * positive).sum(-1)[:, None, :]
    negative_similarity = torch.einsum("btf,bktf->bkt", query, negatives)
    similarities = torch.cat([positive_similarity, negative_similarity], dim=1)

    if valid_mask is None:
        logits = similarities.mean(dim=-1)
    else:
        if valid_mask.shape != (batch_size, frames):
            raise ValueError(
                f"valid_mask must be {(batch_size, frames)}, got {valid_mask.shape}"
            )
        weights = valid_mask[:, None, :].to(similarities.dtype)
        logits = (similarities * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)

    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits / temperature, labels)
