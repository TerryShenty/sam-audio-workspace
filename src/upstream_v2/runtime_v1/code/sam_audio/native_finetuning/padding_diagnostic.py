"""Measure whether right-padding changes SAM Audio's valid-frame training path.

This is a diagnostic, not a trainer.  It compares the same cached target:

1. encoded and forwarded at its native latent length; and
2. right-zero-padded by batching it with a longer cached target.

The timestep and the valid-prefix noise are identical.  Only valid frames of
the shorter target contribute to the reported prediction, loss, and gradient
comparisons.  This directly tests the effect of padding before enabling
variable-length padded batches in formal training.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .benchmark_utils import collate_features, per_target_mse
from .feature_cache import ComponentStore, make_cache_plan
from .model import configure_native_tuning, native_tuning_config_from_mapping
from .objective import make_flow_matching_state
from .optimized_train import CachedFlowModel, model_signature
from .optimized_utils import move_features


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().reshape(-1)
    candidate = candidate.detach().float().reshape(-1)
    difference = candidate - reference
    denominator = reference.norm().clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=0)
    return {
        "cosine": float(cosine.cpu()),
        "relative_l2": float((difference.norm() / denominator).cpu()),
        "max_abs": float(difference.abs().max().cpu()),
        "reference_rms": float(reference.square().mean().sqrt().cpu()),
        "candidate_rms": float(candidate.square().mean().sqrt().cpu()),
    }


def _gradient_group(name: str) -> str:
    if ".lora_" in name:
        return "lora"
    if name.startswith(("memory_proj.", "align_masked_video.", "transformer.y_embedder.")):
        return "condition"
    return "other_trainable"


def _save_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def _gradient_metrics(reference: dict, candidate: dict) -> dict:
    if reference.keys() != candidate.keys():
        raise RuntimeError("Native and padded passes produced different gradient sets")
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    for name in reference:
        a, b = reference[name].double(), candidate[name].double()
        for group in ("all", _gradient_group(name)):
            totals[group][0] += float((a * b).sum())
            totals[group][1] += float((a * a).sum())
            totals[group][2] += float((b * b).sum())
    result = {}
    for group, (dot, aa, bb) in totals.items():
        result[group] = {
            "cosine": dot / max(math.sqrt(aa * bb), 1.0e-30),
            "candidate_to_reference_norm": math.sqrt(bb) / max(math.sqrt(aa), 1.0e-30),
            "reference_norm": math.sqrt(aa),
            "candidate_norm": math.sqrt(bb),
        }
    return result


def _select_pairs(entries: list[dict], count: int) -> list[tuple[dict, dict]]:
    ordered = sorted(entries, key=lambda item: item["latent_frames"])
    candidates = []
    for i, short in enumerate(ordered[:-1]):
        for long in ordered[i + 1 :]:
            if long["latent_frames"] == short["latent_frames"]:
                continue
            ratio = long["latent_frames"] / short["latent_frames"]
            candidates.append((ratio, short, long))
    if not candidates:
        raise RuntimeError("Padding diagnostic needs at least two different latent lengths")
    desired = [1.05, 1.15, 1.30, 1.50, 2.00]
    selected, used = [], set()
    for target_ratio in desired:
        available = [item for item in candidates if (item[1]["sample_id"], item[2]["sample_id"]) not in used]
        if not available or len(selected) >= count:
            break
        ratio, short, long = min(available, key=lambda item: abs(item[0] - target_ratio))
        used.add((short["sample_id"], long["sample_id"]))
        selected.append((short, long))
    return selected


def _load_value(store: ComponentStore, entry: dict) -> dict:
    # Use the first paraphrase consistently; media/GT are identical across variants.
    return store.assemble(entry["variants"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparisons", type=int, default=3)
    parser.add_argument("--time", type=float, default=0.37)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Padding diagnostic requires one CUDA GPU")

    from sam_audio import SAMAudio, SAMAudioProcessor
    from .train import _make_dataset, _resolve_dataset

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if config.get("dtype", "bfloat16") == "bfloat16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
    torch.manual_seed(int(config.get("seed", 2026)))

    model_path = config["model_local_path"]
    sam = SAMAudio.from_pretrained(
        model_path,
        visual_ranker=None,
        text_ranker=None,
        span_predictor=None,
        strict=True,
    )
    processor = SAMAudioProcessor.from_pretrained(model_path)
    sam.to(device=device, dtype=dtype).requires_grad_(False).eval()

    signature = model_signature(model_path, config, processor, sam)
    store = ComponentStore(config["feature_cache_dir"], signature, lru_entries=8)
    root, manifest = _resolve_dataset(config)
    datasets = {
        split: _make_dataset(
            config,
            split=split,
            data_root=root,
            manifest=manifest,
            sample_rate=processor.audio_sampling_rate,
        )
        for split in ("train", "validation")
    }
    plan = make_cache_plan(datasets)
    catalog_path = store.root / f"catalog-{plan['id']}.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Matching cache catalog is missing: {catalog_path}")
    catalog = json.loads(catalog_path.read_text())

    tuning_config = native_tuning_config_from_mapping(
        {**config, "gradient_checkpointing": False, "lora_dropout": 0.0}
    )
    tuning = configure_native_tuning(sam, tuning_config)
    wrapper = CachedFlowModel(sam, float(config.get("sigma_min", 1.0e-5)))
    # Evaluation mode removes dropout while preserving autograd for the selected
    # trainable parameters.  This isolates padding instead of RNG differences.
    wrapper.eval()

    entries = catalog["splits"]["train"]["entries"] + catalog["splits"]["validation"]["entries"]
    pairs = _select_pairs(entries, args.comparisons)
    reports = []

    def autocast():
        return torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)

    for pair_index, (short_entry, long_entry) in enumerate(pairs):
        short_cpu, long_cpu = _load_value(store, short_entry), _load_value(store, long_entry)
        native = move_features(collate_features([short_cpu]), device)
        padded = move_features(collate_features([short_cpu, long_cpu]), device)
        short_frames = native["clean"].shape[1]
        padded_frames = padded["clean"].shape[1]
        if not torch.equal(
            native["forward_args"]["audio_pad_mask"][0],
            padded["forward_args"]["audio_pad_mask"][0, :short_frames],
        ):
            raise RuntimeError("Valid prefix mask changed while padding")
        if padded["forward_args"]["audio_pad_mask"][0, short_frames:].any():
            raise RuntimeError("Right-padding was not masked out")

        generator = torch.Generator(device=device).manual_seed(9000 + pair_index)
        padded_noise = torch.randn(
            padded["clean"].shape,
            generator=generator,
            device=device,
            dtype=padded["clean"].dtype,
        )
        native_noise = padded_noise[:1, :short_frames].clone()
        native_time = torch.tensor([args.time], device=device, dtype=torch.float32)
        padded_time = torch.tensor([args.time, 0.63], device=device, dtype=torch.float32)

        native_state = make_flow_matching_state(
            native["clean"], sigma_min=wrapper.sigma_min, time=native_time, noise=native_noise
        )
        padded_state = make_flow_matching_state(
            padded["clean"], sigma_min=wrapper.sigma_min, time=padded_time, noise=padded_noise
        )

        sam.zero_grad(set_to_none=True)
        with autocast():
            native_prediction = sam(
                noisy_audio=native_state.noisy_latent,
                time=native_state.time,
                **native["forward_args"],
            )
            native_loss = per_target_mse(
                native_prediction,
                native_state.target_velocity,
                native["forward_args"]["audio_pad_mask"],
            )[0]
        native_loss.backward()
        native_gradients = _save_gradients(sam)

        sam.zero_grad(set_to_none=True)
        with autocast():
            padded_prediction = sam(
                noisy_audio=padded_state.noisy_latent,
                time=padded_state.time,
                **padded["forward_args"],
            )
            # Deliberately exclude the longer partner.  The comparison asks only
            # whether padding changes the short target's own learning signal.
            padded_short_loss = per_target_mse(
                padded_prediction[:1],
                padded_state.target_velocity[:1],
                padded["forward_args"]["audio_pad_mask"][:1],
            )[0]
        padded_short_loss.backward()
        padded_gradients = _save_gradients(sam)

        report = {
            "short_sample_id": short_entry["sample_id"],
            "long_sample_id": long_entry["sample_id"],
            "short_duration_seconds": short_entry["duration"],
            "long_duration_seconds": long_entry["duration"],
            "short_latent_frames": short_frames,
            "padded_latent_frames": padded_frames,
            "padding_fraction_of_batch_length": 1.0 - short_frames / padded_frames,
            "prediction_valid_prefix": _tensor_metrics(
                native_prediction[0, :short_frames], padded_prediction[0, :short_frames]
            ),
            "native_loss": float(native_loss.detach().cpu()),
            "padded_masked_loss": float(padded_short_loss.detach().cpu()),
            "loss_relative_change": float(
                ((padded_short_loss - native_loss).abs() / native_loss.abs().clamp_min(1.0e-12)).detach().cpu()
            ),
            "gradients": _gradient_metrics(native_gradients, padded_gradients),
        }
        print(json.dumps(report), flush=True)
        reports.append(report)
        del native, padded, native_prediction, padded_prediction, native_gradients, padded_gradients
        torch.cuda.empty_cache()

    result = {
        "purpose": "native-length versus right-zero-padded valid-prefix parity",
        "model_path": model_path,
        "catalog_path": str(catalog_path),
        "timestep": args.time,
        "tuning": {
            "trainable_parameters": tuning["trainable_parameters"],
            "total_parameters": tuning["total_parameters"],
        },
        "comparisons": reports,
        "interpretation": (
            "Lower relative_l2/loss change and gradient norm change, and cosine values closer to 1, "
            "mean right-padding is less influential. This test does not declare universal thresholds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
