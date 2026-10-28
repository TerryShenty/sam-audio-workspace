"""Trainable wrapper around the released SAM Audio forward pass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from sam_audio.model.model import SAMAudio
from sam_audio.processor import Batch

from .lora import (
    LoRAConfig, apply_sam_audio_lora, apply_sam_audio_ffn_lora, is_lora_parameter,
)
from .objective import flow_matching_loss, make_flow_matching_state


@dataclass
class NativeTuningConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_projections: tuple[str, ...] = ("wq", "wk", "wv", "wo")
    lora_attention_types: tuple[str, ...] = ("attention", "cross_attention")
    ffn_lora_rank: int = 0
    ffn_lora_alpha: float = 32.0
    unfreeze_last_n_blocks: int = 0
    train_memory_proj: bool = True
    train_video_aligner: bool = True
    train_context_embedder: bool = True
    train_output_head: bool = False
    train_input_proj: bool = False
    gradient_checkpointing: bool = True


def native_tuning_config_from_mapping(config: dict[str, Any]) -> NativeTuningConfig:
    return NativeTuningConfig(
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=float(config.get("lora_alpha", 32.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        lora_projections=tuple(
            config.get("lora_projections", ("wq", "wk", "wv", "wo"))
        ),
        lora_attention_types=tuple(
            config.get(
                "lora_attention_types", ("attention", "cross_attention")
            )
        ),
        unfreeze_last_n_blocks=int(config.get("unfreeze_last_n_blocks", 0)),
        ffn_lora_rank=int(config.get("ffn_lora_rank", 0)),
        ffn_lora_alpha=float(config.get("ffn_lora_alpha", 32.0)),
        train_memory_proj=bool(config.get("train_memory_proj", True)),
        train_video_aligner=bool(config.get("train_video_aligner", True)),
        train_context_embedder=bool(config.get("train_context_embedder", True)),
        train_input_proj=bool(config.get("train_input_proj", False)),
        train_output_head=bool(config.get("train_output_head", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
    )


@dataclass
class NativeFineTuneOutput:
    loss: torch.Tensor
    time: torch.Tensor
    prediction_rms: torch.Tensor
    target_velocity_rms: torch.Tensor


def _train_module(module: nn.Module, *, fp32_master: bool) -> None:
    module.requires_grad_(True)
    if fp32_master:
        module.float()


def configure_native_tuning(
    sam_audio: SAMAudio,
    config: NativeTuningConfig,
) -> dict[str, Any]:
    """Freeze the checkpoint, then enable only the requested adaptation paths."""

    sam_audio.requires_grad_(False)
    if config.ffn_lora_rank < 0:
        raise ValueError("ffn_lora_rank must be nonnegative")
    layer_count = len(sam_audio.transformer.layers)
    if not 0 <= config.unfreeze_last_n_blocks <= layer_count:
        raise ValueError(
            "unfreeze_last_n_blocks must be between 0 and " f"{layer_count}"
        )

    first_full_block = layer_count - config.unfreeze_last_n_blocks
    lora_layers = list(range(first_full_block))
    replaced: list[str] = []
    if config.lora_rank > 0 and lora_layers:
        replaced = apply_sam_audio_lora(
            sam_audio.transformer,
            LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                projections=config.lora_projections,
                attention_types=config.lora_attention_types,
            ),
            layer_indices=lora_layers,
        )

    if config.ffn_lora_rank > 0 and lora_layers:
        replaced.extend(apply_sam_audio_ffn_lora(
            sam_audio.transformer,
            rank=config.ffn_lora_rank,
            alpha=config.ffn_lora_alpha,
            dropout=config.lora_dropout,
            layer_indices=lora_layers,
        ))

    # Full-block tuning is an optional stronger ablation. LoRA is omitted from
    # these blocks so the same matrix is not adapted twice. Keep trainable
    # pretrained weights in fp32; autocast still executes eligible matmuls in
    # bf16, while AdamW updates a numerically stable master parameter.
    for block in sam_audio.transformer.layers[first_full_block:]:
        _train_module(block, fp32_master=True)

    if config.train_memory_proj:
        _train_module(sam_audio.memory_proj, fp32_master=True)
    if config.train_video_aligner:
        _train_module(sam_audio.align_masked_video, fp32_master=True)
    if config.train_context_embedder:
        _train_module(sam_audio.transformer.y_embedder, fp32_master=True)
    if config.train_input_proj:
        _train_module(sam_audio.proj, fp32_master=True)
    if config.train_output_head:
        _train_module(sam_audio.transformer.norm, fp32_master=True)
        _train_module(sam_audio.transformer.output, fp32_master=True)
        final_scale_shift = sam_audio.transformer.final_layer_scale_shift_table
        final_scale_shift.data = final_scale_shift.data.float()
        final_scale_shift.requires_grad_(True)

    sam_audio.transformer.gradient_checkpointing = config.gradient_checkpointing

    trainable = {
        name: parameter
        for name, parameter in sam_audio.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("The tuning configuration selected no trainable parameters")
    return {
        "config": asdict(config),
        "lora_modules": replaced,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "total_parameters": sum(parameter.numel() for parameter in sam_audio.parameters()),
        "trainable_names": sorted(trainable),
    }


def trainable_parameter_groups(
    sam_audio: SAMAudio,
    *,
    lora_learning_rate: float,
    condition_learning_rate: float,
    block_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Use a larger LR for new LoRA matrices than pretrained full parameters."""

    groups: dict[str, list[nn.Parameter]] = {
        "lora": [],
        "condition": [],
        "blocks": [],
    }
    seen: set[int] = set()
    for name, parameter in sam_audio.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if is_lora_parameter(name):
            groups["lora"].append(parameter)
        elif name.startswith("memory_proj.") or name.startswith(
            "align_masked_video."
        ) or name.startswith("transformer.y_embedder."):
            groups["condition"].append(parameter)
        else:
            groups["blocks"].append(parameter)

    learning_rates = {
        "lora": lora_learning_rate,
        "condition": condition_learning_rate,
        "blocks": block_learning_rate,
    }
    return [
        {
            "params": parameters,
            "lr": learning_rates[name],
            "weight_decay": weight_decay,
            "group_name": name,
        }
        for name, parameters in groups.items()
        if parameters
    ]


class NativeSAMAudioFineTuner(nn.Module):
    """Compute native SAM Audio flow loss without invoking inference/reranking."""

    def __init__(
        self,
        sam_audio: SAMAudio,
        *,
        sigma_min: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.sam_audio = sam_audio
        self.sigma_min = sigma_min

    def train(self, mode: bool = True):
        super().train(mode)
        # Only modules participating in the trainable path use train mode. The
        # frozen encoders stay deterministic; DiT dropout remains active.
        self.sam_audio.eval()
        self.sam_audio.transformer.train(mode)
        for module in (
            self.sam_audio.memory_proj,
            self.sam_audio.align_masked_video,
            self.sam_audio.transformer.y_embedder,
        ):
            if any(parameter.requires_grad for parameter in module.parameters()):
                module.train(mode)
        return self

    def _encode_joint_target(
        self,
        target: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_target = self.sam_audio.audio_codec(target).transpose(1, 2)
            z_residual = self.sam_audio.audio_codec(residual).transpose(1, 2)
        return torch.cat([z_target, z_residual], dim=-1)

    def forward(
        self,
        *,
        condition_batch: Batch,
        target: torch.Tensor,
        residual: torch.Tensor,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> NativeFineTuneOutput:
        with torch.no_grad():
            forward_args = self.sam_audio._get_forward_args(condition_batch)
            clean_joint = self._encode_joint_target(target, residual)

        if clean_joint.shape != forward_args["audio_features"].shape:
            raise RuntimeError(
                "Mixture and target/residual encoded to different shapes: "
                f"mixture={forward_args['audio_features'].shape}, "
                f"target_residual={clean_joint.shape}"
            )
        state = make_flow_matching_state(
            clean_joint,
            sigma_min=self.sigma_min,
            time=time,
            noise=noise,
        )
        prediction = self.sam_audio(
            noisy_audio=state.noisy_latent,
            time=state.time,
            **forward_args,
        )
        loss = flow_matching_loss(
            prediction,
            state.target_velocity,
            forward_args["audio_pad_mask"],
        )
        return NativeFineTuneOutput(
            loss=loss,
            time=state.time.detach(),
            prediction_rms=prediction.detach().float().square().mean().sqrt(),
            target_velocity_rms=state.target_velocity.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.sam_audio.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict: bool = True
    ) -> None:
        current = dict(self.sam_audio.named_parameters())
        missing = sorted(
            name
            for name, parameter in current.items()
            if parameter.requires_grad and name not in state_dict
        )
        unexpected = sorted(
            name for name in state_dict
            if name not in current or not current[name].requires_grad
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Trainable checkpoint mismatch; missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        with torch.no_grad():
            for name, value in state_dict.items():
                if name in current:
                    current[name].copy_(value.to(current[name]))
