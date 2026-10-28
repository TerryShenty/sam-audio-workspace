"""Trainable AUDSEG adapter around a frozen public SAM Audio checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchdiffeq import odeint

from sam_audio.model.model import DFLT_ODE_OPT, SAMAudio, SeparationResult

from .losses import (
    cosine_alignment_loss,
    flow_matching_loss,
    referring_contrastive_loss,
    sample_flow_matching_state,
)
from .vlm import QwenAudsegEncoder


class AudsegAdapter(nn.Module):
    """Map a VLM hidden vector into the released T5 token space."""

    def __init__(self, input_dim: int, output_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).unsqueeze(1)


class AlignmentHead(nn.Module):
    """Three-layer MLP following the architecture stated in the SAM Audio paper."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


@dataclass
class ReferringTrainingOutput:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    flow_loss: torch.Tensor
    alignment_loss: torch.Tensor
    referring_loss: torch.Tensor
    audseg_accuracy: torch.Tensor
    time: torch.Tensor


class ReferringSAMAudio(nn.Module):
    """First-stage referring fine-tuning while the released backbone stays frozen.

    The default audio teacher is the frozen DAC-VAE latent.  It preserves the
    paper's frame-level ``1-cosine`` objective and makes the implementation
    runnable using public weights only.  It is not claimed to reproduce Meta's
    unpublished frame-level PANNs/AED extraction exactly.  The forward method also
    accepts externally precomputed AED features to replace this fallback.
    """

    def __init__(
        self,
        sam_audio: SAMAudio,
        vlm: QwenAudsegEncoder,
        *,
        hidden_layer: int,
        adapter_hidden_dim: int = 1024,
        teacher_dim: int = 128,
        alignment_hidden_dim: int = 2048,
        sigma_min: float = 1.0e-5,
        ce_weight: float = 0.1,
        alignment_weight: float = 1.0,
        referring_weight: float = 1.0,
        referring_temperature: float = 0.07,
        train_memory_proj: bool = False,
        train_sam_modules: tuple[str, ...] = (),
    ):
        super().__init__()
        self.sam_audio = sam_audio
        self.vlm = vlm
        self.hidden_layer = hidden_layer
        self.sigma_min = sigma_min
        self.ce_weight = ce_weight
        self.alignment_weight = alignment_weight
        self.referring_weight = referring_weight
        self.referring_temperature = referring_temperature
        self.teacher_dim = teacher_dim

        text_dim = sam_audio.memory_proj.in_features
        dit_dim = sam_audio.transformer.output.in_features
        self.audseg_adapter = AudsegAdapter(
            vlm.hidden_size, text_dim, adapter_hidden_dim
        )
        self.alignment_head = AlignmentHead(dit_dim, teacher_dim, alignment_hidden_dim)

        sam_audio.requires_grad_(False).eval()
        if train_memory_proj:
            sam_audio.memory_proj.requires_grad_(True)
        # One-stage joint optimization: enable selected existing SAM modules now,
        # before the optimizer is built; no adapter-only warmup phase.
        self.train_sam_modules = tuple(train_sam_modules)
        for name in self.train_sam_modules:
            if not name or name.split(".")[0] in {"audio_codec", "text_encoder", "vision_encoder", "visual_encoder"}:
                raise ValueError(f"Refuse to unfreeze a frozen encoder: {name!r}")
            sam_audio.get_submodule(name).requires_grad_(True)
        self.train_memory_proj = train_memory_proj

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen components stay deterministic but still participate in autograd;
        # do not put the DiT forward inside torch.no_grad().
        self.sam_audio.eval()
        self.vlm.train(mode)
        self.audseg_adapter.train(mode)
        self.alignment_head.train(mode)
        if self.train_memory_proj:
            self.sam_audio.memory_proj.train(mode)
        for name in self.train_sam_modules:
            self.sam_audio.get_submodule(name).train(mode)
        return self

    @staticmethod
    def _match_frames(features: torch.Tensor, frames: int) -> torch.Tensor:
        if features.size(-2) == frames:
            return features
        leading_shape = features.shape[:-2]
        feature_dim = features.size(-1)
        flat = features.reshape(-1, features.size(-2), feature_dim).transpose(1, 2)
        flat = torch.nn.functional.interpolate(
            flat, size=frames, mode="linear", align_corners=False
        )
        return flat.transpose(1, 2).reshape(*leading_shape, frames, feature_dim)

    def _encode_audio_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        # DAC-VAE's public encoder is already no-grad; explicit detach documents that
        # it is a fixed target representation rather than a learned escape route.
        return self.sam_audio.audio_codec(waveform).transpose(1, 2).detach()

    def _text_condition(
        self, video_paths: list[str], prompts: list[str],
        target_descriptions: list[str | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t5_features, t5_mask = self.sam_audio.text_encoder(prompts)
        vlm_output = (self.vlm(video_paths, prompts) if target_descriptions is None
                      else self.vlm(video_paths, prompts, target_descriptions=target_descriptions))
        adapter_dtype = next(self.audseg_adapter.parameters()).dtype
        audseg = self.audseg_adapter(vlm_output.hidden_state.to(adapter_dtype))
        audseg = audseg.to(dtype=t5_features.dtype, device=t5_features.device)
        text_features = torch.cat([t5_features, audseg], dim=1)
        audseg_mask = torch.ones(
            t5_mask.size(0), 1, dtype=torch.bool, device=t5_mask.device
        )
        return (
            text_features,
            torch.cat([t5_mask, audseg_mask], dim=1),
            vlm_output.ce_loss,
            vlm_output.token_accuracy,
        )

    def forward(
        self,
        *,
        mixture: torch.Tensor,
        target: torch.Tensor,
        residual: torch.Tensor,
        negatives: torch.Tensor,
        video_paths: list[str],
        prompts: list[str],
        target_descriptions: list[str | None] | None = None,
        masked_video_features: torch.Tensor | None = None,
        positive_teacher_features: torch.Tensor | None = None,
        negative_teacher_features: torch.Tensor | None = None,
    ) -> ReferringTrainingOutput:
        """Compute CE, flow, and referring losses with an optional alignment term."""

        if mixture.ndim != 3 or mixture.size(1) != 1:
            raise ValueError("mixture/target/residual must have shape [B, 1, samples]")
        if negatives.ndim != 4 or negatives.size(2) != 1:
            raise ValueError("negatives must have shape [B, K, 1, samples]")

        batch_size, negative_count = negatives.shape[:2]
        z_mix = self._encode_audio_latent(mixture)
        z_target = self._encode_audio_latent(target)
        z_residual = self._encode_audio_latent(residual)
        joint_clean = torch.cat([z_target, z_residual], dim=-1)
        flow_state = sample_flow_matching_state(joint_clean, self.sigma_min)

        text_features, text_mask, loss_ce, audseg_accuracy = self._text_condition(
            video_paths, prompts, target_descriptions
        )
        audio_features = torch.cat([z_mix, z_mix], dim=-1)
        audio_pad_mask = torch.ones(
            batch_size,
            audio_features.size(1),
            dtype=torch.bool,
            device=audio_features.device,
        )

        velocity, hidden = self.sam_audio(
            noisy_audio=flow_state.noisy_latent,
            audio_features=audio_features,
            text_features=text_features,
            time=flow_state.time,
            text_mask=text_mask,
            audio_pad_mask=audio_pad_mask,
            return_hidden_layer=self.hidden_layer,
            masked_video_features=masked_video_features,
        )
        loss_flow = flow_matching_loss(
            velocity, flow_state.target_velocity, audio_pad_mask
        )

        predicted_teacher = self.alignment_head(hidden)
        if positive_teacher_features is None:
            if self.teacher_dim != z_target.size(-1):
                raise ValueError(
                    "teacher_dim differs from DAC latent dimension; provide "
                    "positive_teacher_features and negative_teacher_features"
                )
            positive_teacher_features = z_target
        positive_teacher_features = self._match_frames(
            positive_teacher_features.detach(), predicted_teacher.size(1)
        )
        loss_alignment = cosine_alignment_loss(
            predicted_teacher, positive_teacher_features, audio_pad_mask
        )

        if negative_teacher_features is None:
            flat_negatives = negatives.reshape(
                batch_size * negative_count, 1, negatives.size(-1)
            )
            negative_teacher_features = self._encode_audio_latent(flat_negatives)
            negative_teacher_features = negative_teacher_features.reshape(
                batch_size,
                negative_count,
                negative_teacher_features.size(1),
                negative_teacher_features.size(2),
            )
        negative_teacher_features = self._match_frames(
            negative_teacher_features.detach(), predicted_teacher.size(1)
        )
        loss_referring = referring_contrastive_loss(
            predicted_teacher,
            positive_teacher_features,
            negative_teacher_features,
            audio_pad_mask,
            self.referring_temperature,
        )
        total = (
            self.ce_weight * loss_ce
            + loss_flow
            + self.alignment_weight * loss_alignment
            + self.referring_weight * loss_referring
        )
        return ReferringTrainingOutput(
            loss=total,
            ce_loss=loss_ce.detach(),
            flow_loss=loss_flow.detach(),
            alignment_loss=loss_alignment.detach(),
            referring_loss=loss_referring.detach(),
            audseg_accuracy=audseg_accuracy.detach(),
            time=flow_state.time.detach(),
        )

    def trainable_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Save only additions (and optionally memory_proj), never the 3B backbone."""

        state = {
            "vlm": self.vlm.trainable_state_dict(),
            "audseg_adapter": self.audseg_adapter.state_dict(),
            "alignment_head": self.alignment_head.state_dict(),
        }
        if self.train_memory_proj:
            state["memory_proj"] = self.sam_audio.memory_proj.state_dict()
        if self.train_sam_modules:
            state["sam_modules"] = {
                name: self.sam_audio.get_submodule(name).state_dict()
                for name in self.train_sam_modules
            }
        return state

    def load_trainable_state_dict(
        self, state: dict[str, dict[str, torch.Tensor]], strict: bool = True
    ) -> None:
        saved_modules = state.get("sam_modules", {})
        if strict and set(saved_modules) != set(self.train_sam_modules):
            raise ValueError("Checkpoint SAM module selection differs from this model")
        for name in self.train_sam_modules:
            if name in saved_modules:
                self.sam_audio.get_submodule(name).load_state_dict(saved_modules[name], strict=strict)
        if "vlm" not in state:
            if strict:
                raise KeyError("Checkpoint is missing trainable VLM/LoRA state")
        else:
            self.vlm.load_trainable_state_dict(state["vlm"])
        self.audseg_adapter.load_state_dict(state["audseg_adapter"], strict=strict)
        self.alignment_head.load_state_dict(state["alignment_head"], strict=strict)
        if "memory_proj" in state:
            self.sam_audio.memory_proj.load_state_dict(
                state["memory_proj"], strict=strict
            )

    @torch.inference_mode()
    def separate_referring(
        self,
        *,
        mixture: torch.Tensor,
        video_paths: list[str],
        prompts: list[str],
        noise: torch.Tensor | None = None,
        masked_video_features: torch.Tensor | None = None,
        ode_opt: dict | None = None,
    ) -> SeparationResult:
        """Run the released ODE sampler with T5 + the learned AUDSEG condition."""

        self.eval()
        original_samples = mixture.size(-1)
        z_mix = self._encode_audio_latent(mixture)
        audio_features = torch.cat([z_mix, z_mix], dim=-1)
        text_features, text_mask, _, _ = self._text_condition(video_paths, prompts)
        audio_pad_mask = torch.ones(
            mixture.size(0),
            audio_features.size(1),
            dtype=torch.bool,
            device=mixture.device,
        )
        if noise is None:
            noise = torch.randn_like(audio_features)

        def vector_field(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self.sam_audio(
                noisy_audio=state,
                audio_features=audio_features,
                text_features=text_features,
                time=time.expand(state.size(0)),
                text_mask=text_mask,
                audio_pad_mask=audio_pad_mask,
                masked_video_features=masked_video_features,
            )

        states = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=mixture.device),
            **(ode_opt or DFLT_ODE_OPT),
        )
        generated = states[-1].transpose(1, 2)
        batch_size, channels, frames = generated.shape
        decoded = self.sam_audio.audio_codec.decode(
            generated.reshape(batch_size * 2, channels // 2, frames)
        )
        decoded = decoded.view(batch_size, 2, -1)[..., :original_samples]
        return SeparationResult(
            target=decoded[:, 0],
            residual=decoded[:, 1],
            noise=noise,
        )
