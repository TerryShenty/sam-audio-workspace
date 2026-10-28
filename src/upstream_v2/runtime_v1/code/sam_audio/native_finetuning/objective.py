"""The conditional flow-matching objective described in the SAM Audio paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowMatchingState:
    noisy_latent: torch.Tensor
    target_velocity: torch.Tensor
    time: torch.Tensor
    noise: torch.Tensor


def make_flow_matching_state(
    clean_latent: torch.Tensor,
    *,
    sigma_min: float = 1.0e-5,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> FlowMatchingState:
    """Sample ``x_t`` and its constant conditional target velocity.

    This formula is channel-agnostic. The public joint checkpoint uses
    ``concat(z_target, z_residual)`` with shape ``[B, T, 256]``; the converted
    target-only path passes only ``z_target`` with shape ``[B, T, 128]``.
    """

    if clean_latent.ndim != 3:
        raise ValueError(
            f"clean_latent must have shape [B, T, C], got {clean_latent.shape}"
        )
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("sigma_min must be in [0, 1)")
    batch_size = clean_latent.size(0)
    if time is None:
        time = torch.rand(batch_size, device=clean_latent.device, dtype=torch.float32)
    if time.shape != (batch_size,):
        raise ValueError(f"time must have shape [{batch_size}], got {time.shape}")
    if noise is None:
        noise = torch.randn_like(clean_latent)
    if noise.shape != clean_latent.shape:
        raise ValueError(
            f"noise must match clean_latent: {noise.shape} vs {clean_latent.shape}"
        )

    t = time.to(clean_latent.dtype)[:, None, None]
    noisy_latent = (1.0 - (1.0 - sigma_min) * t) * noise + t * clean_latent
    target_velocity = clean_latent - (1.0 - sigma_min) * noise
    return FlowMatchingState(noisy_latent, target_velocity, time, noise)


def flow_matching_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frame-masked MSE, accumulated in fp32 for bf16 training stability."""

    if prediction.shape != target_velocity.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs "
            f"{target_velocity.shape}"
        )
    squared_error = (prediction.float() - target_velocity.float()).square()
    if valid_mask is None:
        return squared_error.mean()
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {prediction.shape[:2]}, got {valid_mask.shape}"
        )
    weights = valid_mask.to(squared_error.dtype).unsqueeze(-1)
    denominator = weights.sum() * squared_error.size(-1)
    return (squared_error * weights).sum() / denominator.clamp_min(1.0)
