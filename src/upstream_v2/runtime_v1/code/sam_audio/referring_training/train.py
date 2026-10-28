"""Single-node training entry point for the first referring baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sam_audio import SAMAudio

from .data import ReferringAudioDataset, collate_referring_batch
from .model import ReferringSAMAudio
from .vlm import QwenAudsegEncoder


def _torch_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError(f"Unsupported dtype {name}; choose from {sorted(choices)}")
    return choices[name]


def _default_hidden_layer(layer_count: int) -> int:
    # Paper: 2nd/4th/6th block for 500M/1B/3B respectively.
    mapping = {12: 1, 16: 3, 22: 5}
    if layer_count not in mapping:
        raise ValueError(
            f"Unknown {layer_count}-layer SAM Audio variant; set hidden_layer explicitly"
        )
    return mapping[layer_count]


def _to_device(batch: dict, device: torch.device) -> dict:
    for key in ("mixture", "target", "residual", "negatives"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _save_checkpoint(
    path: Path,
    model: ReferringSAMAudio,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": config,
            "trainable_model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def train(config: dict) -> None:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(config.get("device", "cuda"))
    dtype = _torch_dtype(config.get("dtype", "bfloat16"))
    sam_audio = SAMAudio.from_pretrained(config["sam_audio_model"])
    sam_audio.to(device=device, dtype=dtype).eval()

    vlm = QwenAudsegEncoder(
        config.get("vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct"),
        device=device,
        dtype=dtype,
        backend=config.get("vlm_backend", "qwen2_5_vl"),
        fps=float(config.get("vlm_fps", 1.0)),
        min_pixels=config.get("vlm_min_pixels"),
        max_pixels=config.get("vlm_max_pixels"),
        attn_implementation=config.get("vlm_attn_implementation"),
        lora_rank=int(config.get("vlm_lora_rank", 8)),
        lora_alpha=int(config.get("vlm_lora_alpha", 16)),
        lora_dropout=float(config.get("vlm_lora_dropout", 0.05)),
        lora_target_modules=tuple(
            config.get("vlm_lora_target_modules", ["q_proj", "v_proj"])
        ),
        gradient_checkpointing=bool(config.get("vlm_gradient_checkpointing", True)),
    )
    hidden_layer = config.get("hidden_layer")
    if hidden_layer is None:
        hidden_layer = _default_hidden_layer(len(sam_audio.transformer.layers))

    model = ReferringSAMAudio(
        sam_audio,
        vlm,
        hidden_layer=int(hidden_layer),
        adapter_hidden_dim=int(config.get("adapter_hidden_dim", 1024)),
        teacher_dim=int(config.get("teacher_dim", 128)),
        alignment_hidden_dim=int(config.get("alignment_hidden_dim", 2048)),
        sigma_min=float(config.get("sigma_min", 1.0e-5)),
        ce_weight=float(config.get("ce_weight", 0.1)),
        alignment_weight=float(config.get("alignment_weight", 0.0)),
        referring_weight=float(config.get("referring_weight", 0.1)),
        referring_temperature=float(config.get("referring_temperature", 0.07)),
        train_memory_proj=bool(config.get("train_memory_proj", False)),
        train_sam_modules=tuple(config.get("train_sam_modules", [])),
    )
    model.audseg_adapter.to(device)
    model.alignment_head.to(device)
    model.train()

    dataset = ReferringAudioDataset(
        config["prompt_manifest"],
        target_manifest=config.get("target_manifest"),
        data_root=config.get("data_root"),
        sample_rate=sam_audio.sample_rate,
        clip_seconds=float(config.get("clip_seconds", 10.0)),
        num_negatives=int(config.get("num_negatives", 1)),
        seed=seed,
        require_negatives=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_referring_batch,
    )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable parameters were selected")
    vlm_parameter_ids = {
        id(parameter) for parameter in model.vlm.parameters() if parameter.requires_grad
    }
    vlm_trainable = [
        parameter for parameter in trainable if id(parameter) in vlm_parameter_ids
    ]
    other_trainable = [
        parameter for parameter in trainable if id(parameter) not in vlm_parameter_ids
    ]
    learning_rate = float(config.get("learning_rate", 1.0e-4))
    vlm_learning_rate = float(config.get("vlm_learning_rate", learning_rate))
    parameter_groups = []
    if other_trainable:
        parameter_groups.append(
            {"params": other_trainable, "lr": learning_rate, "name": "new_modules"}
        )
    if vlm_trainable:
        parameter_groups.append(
            {"params": vlm_trainable, "lr": vlm_learning_rate, "name": "vlm"}
        )
    print(
        json.dumps(
            {
                "trainable_parameters": sum(p.numel() for p in trainable),
                "vlm_trainable_parameters": sum(p.numel() for p in vlm_trainable),
                "new_module_parameters": sum(p.numel() for p in other_trainable),
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    max_steps = int(config.get("max_steps", 10_000))
    # Learning-rate warmup applies to ALL groups, not an adapter-only phase.
    warmup_steps = int(config.get("warmup_steps", 500))

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    grad_accumulation = int(config.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    log_every = int(config.get("log_every", 10))
    save_every = int(config.get("save_every", 500))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    while global_step < max_steps:
        for batch in loader:
            batch = _to_device(batch, device)
            autocast_enabled = device.type == "cuda" and dtype != torch.float32
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                output = model(
                    mixture=batch["mixture"],
                    target=batch["target"],
                    residual=batch["residual"],
                    negatives=batch["negatives"],
                    video_paths=batch["video_path"],
                    prompts=batch["prompt"],
                    target_descriptions=(batch["target_description"]
                        if config.get("supervise_target_description", False) else None),
                )
                loss = output.loss / grad_accumulation
            loss.backward()
            micro_step += 1
            if micro_step % grad_accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                record = {
                    "step": global_step,
                    "loss": float(output.loss.detach()),
                    "ce": float(output.ce_loss),
                    "flow": float(output.flow_loss),
                    "alignment": float(output.alignment_loss),
                    "referring": float(output.referring_loss),
                    "audseg_accuracy": float(output.audseg_accuracy),
                    "lr": scheduler.get_last_lr(),
                }
                print(json.dumps(record), flush=True)
            if global_step % save_every == 0:
                _save_checkpoint(
                    output_dir / f"step_{global_step:07d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                )
            if global_step >= max_steps:
                break

    _save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        global_step,
        config,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    train(config)


if __name__ == "__main__":
    main()
