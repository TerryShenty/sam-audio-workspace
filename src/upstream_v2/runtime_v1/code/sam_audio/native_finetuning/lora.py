"""Small, explicit LoRA implementation for SAM Audio's custom DiT.

The released DiT does not inherit from ``transformers.PreTrainedModel`` and its
attention projections are named ``wq/wk/wv/wo``.  Keeping the adapter explicit
makes the exact matrices being trained auditable and avoids depending on model
name heuristics in a generic PEFT wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)

        # Keep the trainable master weights in fp32. Autocast performs the
        # matmuls in bf16 while AdamW updates numerically stable fp32 weights.
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False).float()
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False).float()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.dropout(x)))
        return base + update * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    projections: tuple[str, ...] = ("wq", "wk", "wv", "wo")
    attention_types: tuple[str, ...] = ("attention", "cross_attention")


def _selected_layer_indices(
    layer_count: int,
    layer_indices: Iterable[int] | None,
) -> set[int]:
    selected = set(range(layer_count)) if layer_indices is None else set(layer_indices)
    invalid = sorted(index for index in selected if not 0 <= index < layer_count)
    if invalid:
        raise ValueError(
            f"LoRA layer indices {invalid} are outside [0, {layer_count - 1}]"
        )
    return selected


def apply_sam_audio_lora(
    transformer: nn.Module,
    config: LoRAConfig,
    *,
    layer_indices: Iterable[int] | None = None,
) -> list[str]:
    """Replace selected DiT attention projections and return their full names.

    ``attention`` is self-attention over the noisy audio sequence;
    ``cross_attention`` reads the text memory.  Visual features are added to the
    audio sequence before the DiT, so adapting both attention paths is the
    conservative first-stage setting.
    """

    layers = transformer.layers
    selected = _selected_layer_indices(len(layers), layer_indices)
    valid_attention_types = {"attention", "cross_attention"}
    invalid_attention_types = set(config.attention_types) - valid_attention_types
    if invalid_attention_types:
        raise ValueError(
            f"Unknown attention types: {sorted(invalid_attention_types)}"
        )
    valid_projections = {"wq", "wk", "wv", "wo"}
    invalid_projections = set(config.projections) - valid_projections
    if invalid_projections:
        raise ValueError(f"Unknown attention projections: {sorted(invalid_projections)}")

    replaced: list[str] = []
    for layer_index, block in enumerate(layers):
        if layer_index not in selected:
            continue
        for attention_name in config.attention_types:
            attention = getattr(block, attention_name)
            if attention is None:
                continue
            for projection_name in config.projections:
                base = getattr(attention, projection_name)
                if isinstance(base, LoRALinear):
                    raise ValueError(
                        "LoRA was already applied to "
                        f"layers.{layer_index}.{attention_name}.{projection_name}"
                    )
                if not isinstance(base, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at {attention_name}.{projection_name}, "
                        f"found {type(base).__name__}"
                    )
                wrapped = LoRALinear(
                    base,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ).to(device=base.weight.device)
                setattr(attention, projection_name, wrapped)
                replaced.append(
                    f"transformer.layers.{layer_index}.{attention_name}."
                    f"{projection_name}"
                )
    if not replaced:
        raise RuntimeError("No DiT projection was selected for LoRA")
    return replaced


def is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def apply_sam_audio_ffn_lora(
    transformer: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    layer_indices: Iterable[int] | None = None,
) -> list[str]:
    """Adapt the SwiGLU FFN projections without changing the hidden/state sizes.

    The optional w3 exists only for SwiGLU. Full-tuned blocks are excluded by
    configure_native_tuning, just as for attention LoRA.
    """
    selected = _selected_layer_indices(len(transformer.layers), layer_indices)
    candidates = []
    for index, block in enumerate(transformer.layers):
        if index not in selected:
            continue
        ffn = block.feed_forward
        names = ("w1", "w2", "w3") if ffn.swiglu else ("w1", "w2")
        for name in names:
            base = getattr(ffn, name)
            if isinstance(base, LoRALinear):
                raise ValueError(f"LoRA was already applied to layers.{index}.feed_forward.{name}")
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected nn.Linear at layers.{index}.feed_forward.{name}")
            candidates.append((index, ffn, name, base))
    if not candidates:
        raise RuntimeError("No DiT FFN projection was selected for LoRA")
    replaced = []
    for index, ffn, name, base in candidates:
        wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        setattr(ffn, name, wrapped.to(device=base.weight.device))
        replaced.append(f"transformer.layers.{index}.feed_forward.{name}")
    return replaced


@torch.no_grad()
def merge_lora_modules(module: nn.Module) -> list[str]:
    """Fold every LoRA delta into its base weight for ordinary inference."""

    merged: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                delta = child.lora_B.weight.float() @ child.lora_A.weight.float()
                child.base_layer.weight.add_(
                    (delta * child.scaling).to(child.base_layer.weight)
                )
                setattr(parent, child_name, child.base_layer)
                merged.append(full_name)
            else:
                visit(child, full_name)

    visit(module, "")
    return merged
