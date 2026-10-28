"""Load a delta checkpoint back into the public SAM Audio inference API."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sam_audio import SAMAudio

from .lora import merge_lora_modules
from .checkpoint import require_matching_base
from .model import (
    NativeSAMAudioFineTuner,
    configure_native_tuning,
    native_tuning_config_from_mapping,
)


def load_finetuned_sam_audio(
    checkpoint_path: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    load_inference_helpers: bool = False,
) -> SAMAudio:
    """Return a normal ``SAMAudio`` model with the fine-tuned delta merged.

    Set ``load_inference_helpers=True`` only when span prediction or candidate
    reranking will actually be used; they are unnecessary for the direct
    one-candidate validation that should measure the backbone change first.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not checkpoint.get("verified_base"):
        raise ValueError("Legacy delta lacks verified pretrained base identity; do not reuse the broken smoke delta")
    model_id = model_id or config.get("model_local_path") or config["sam_audio_model"]
    helper_overrides = (
        {}
        if load_inference_helpers
        else {"visual_ranker": None, "text_ranker": None, "span_predictor": None}
    )
    sam_audio = SAMAudio.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        revision=revision if revision is not None else config.get("model_revision"),
        strict=True,
        **helper_overrides,
    )
    if config.get("target_only",False):
        from .target_only import convert_target_only
        convert_target_only(sam_audio)
    require_matching_base(checkpoint, sam_audio)
    # Optimized training rounded the frozen base to its training dtype before
    # creating fp32 trainable masters. Recreate that base before applying delta.
    training_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.get("dtype", "bfloat16")]
    sam_audio.to(device=device, dtype=training_dtype)
    tuning = native_tuning_config_from_mapping(config)
    configure_native_tuning(sam_audio, tuning)
    wrapper = NativeSAMAudioFineTuner(
        sam_audio, sigma_min=float(config.get("sigma_min", 1.0e-5))
    )
    wrapper.load_trainable_state_dict(checkpoint["trainable_model"])
    # Merge in fp32 so the learned delta is not rounded away into a bf16 base.
    sam_audio.float()
    merge_lora_modules(sam_audio.transformer)
    sam_audio.requires_grad_(False)
    # Condition modules used fp32 master weights during training. The default
    # stays fp32 because the public processor returns fp32 waveforms and the
    # public separate() method does not enter autocast itself. Callers choosing
    # bf16 should wrap separate() in torch.autocast("cuda", torch.bfloat16).
    return sam_audio.to(device=device, dtype=dtype).eval()
