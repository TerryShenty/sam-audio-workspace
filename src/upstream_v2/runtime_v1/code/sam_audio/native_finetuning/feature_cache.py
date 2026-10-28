"""Component-deduplicated frozen features; no trainable projections are cached.

Uses released SAM Audio encoder/processor methods. MMAudio's pre-extracted
conditioning pattern is the reference, NOT its VAE sampling distribution:
SAM Audio's public DAC encoder returns the posterior mean, which we preserve.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .benchmark_utils import collate_features
from .optimized_utils import atomic_json, digest, select_variant


class ComponentStore:
    def __init__(self, directory, signature, lru_entries=0):
        self.signature = signature
        self.root = Path(directory) / digest(signature)
        self.lru_entries = lru_entries
        self.memory = OrderedDict()
        self.counts = {}

    def path(self, kind, key):
        return self.root / kind / key[:2] / (key + ".pt")

    def exists(self, kind, key):
        return self.path(kind, key).is_file()

    def put(self, kind, key, value):
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(value, temporary)
        temporary.replace(path)
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def get(self, kind, key):
        ident = (kind, key)
        if ident in self.memory:
            self.memory.move_to_end(ident)
            return self.memory[ident]
        value = torch.load(self.path(kind, key), map_location="cpu", weights_only=True)
        if self.lru_entries:
            self.memory[ident] = value
            if len(self.memory) > self.lru_entries:
                self.memory.popitem(last=False)
        return value

    def assemble(self, refs):
        mixture = self.get("mixture", refs["mixture"])
        args = dict(mixture["forward_args"])
        args["masked_video_features"] = self.get("video", refs["video"])
        args.update(self.get("text", refs["text"]))
        clean = self.get("clean", refs["clean"])
        target_only = "conversion" in self.signature.get("verified_base", {})
        expected = (*args["audio_features"].shape[:-1],128) if target_only else args["audio_features"].shape
        if clean.shape != expected:
            raise ValueError("Cached target and mixture latent shapes disagree")
        return {"forward_args": args, "clean": clean}


def make_cache_plan(datasets):
    """Media fingerprints detect ordinary in-place edits; model/preproc is a namespace.

    Paths + sizes + nanosecond mtimes are intentional (not a content hash).
    Pin dataset revision for published experiments; do not preserve mtimes while
    replacing media. New manifest rows reuse unchanged component objects.
    """
    fingerprints = {}

    def identity(path):
        path = Path(path).resolve()
        if str(path) not in fingerprints:
            stat = path.stat()
            fingerprints[str(path)] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        return fingerprints[str(path)]

    plan = {"mixtures": {}, "texts": {}, "splits": {}}
    for split, dataset in datasets.items():
        entries = []
        for rows in dataset.examples:
            variants = []
            for row in rows:
                if getattr(dataset, "require_explicit_mixture_path", False) and not row.get("mixture_audio_path", "").strip():
                    raise ValueError("Explicit mixture_audio_path is required; do not silently substitute video audio")

                def path(field, fallback=None):
                    value = row.get(field) or (row.get(fallback) if fallback else None)
                    if not value:
                        return None
                    p = Path(value)
                    return p if p.is_absolute() else dataset.data_root / p

                mix_path, video_path = (
                    path("mixture_audio_path", "video_path"),
                    path("video_path"),
                )
                target_path, residual_path = (
                    path("gt_audio_path"),
                    None if getattr(dataset,"target_only",False) else path("residual_audio_path"),
                )
                mix_key = digest(
                    [identity(mix_path), dataset.max_samples, dataset.sample_rate]
                )
                video_key = digest(
                    [identity(video_path), mix_key]
                )  # alignment depends on mixture length
                clean_key = digest(
                    [
                        identity(target_path),
                        mix_key,
                        identity(residual_path)
                        if residual_path
                        else ("target-only-v1" if getattr(dataset,"target_only",False) else "mixture-minus-target"),
                    ]
                )
                prompt = row["prompt"].strip()
                text_key = digest(prompt)
                mix = plan["mixtures"].setdefault(
                    mix_key,
                    {
                        "path": str(mix_path),
                        "max_samples": dataset.max_samples,
                        "videos": {},
                        "targets": {},
                    },
                )
                mix["videos"][video_key] = str(video_path)
                mix["targets"][clean_key] = {
                    "target": str(target_path),
                    "target_only": getattr(dataset,"target_only",False),
                    "residual": str(residual_path) if residual_path else None,
                }
                plan["texts"][text_key] = prompt
                variants.append(
                    {
                        "mixture": mix_key,
                        "video": video_key,
                        "clean": clean_key,
                        "text": text_key,
                        "prompt": prompt,
                    }
                )
            # Sampling a paraphrase may not silently switch the GT/video.
            if len({(v["mixture"], v["video"], v["clean"]) for v in variants}) != 1:
                raise ValueError(
                    "One target group has conflicting media across paraphrases"
                )
            row = rows[0]
            entries.append(
                {
                    "sample_id": f"{row.get('dataset', '')}::{row['clip_id']}::{row['target_source_id']}",
                    "variants": variants,
                }
            )
        plan["splits"][split] = {"seed": dataset.seed, "entries": entries}
    plan["id"] = digest(plan)
    return plan


class FrozenExtractor:
    def __init__(
        self,
        sam,
        processor,
        device,
        dtype,
        timer,
        *,
        to_cpu=False,
        video_frame_chunk=0,
    ):
        self.sam, self.processor, self.device = sam, processor, torch.device(device)
        self.dtype, self.timer, self.to_cpu = dtype, timer, to_cpu
        self.video_frame_chunk = int(video_frame_chunk)
        # Also applies to the independent sam._get_forward_args parity path.
        # Do not persist first-call TorchScript profiling output as a label.
        sam.audio_codec._stable_encoding = True
        for module in (sam.audio_codec, sam.text_encoder, sam.vision_encoder):
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError(
                    "Frozen feature extraction cannot cache trainable encoders"
                )
            module.eval()

    def finish(self, value):
        if isinstance(value, dict):
            return {k: self.finish(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            return value.detach().cpu() if self.to_cpu else value.detach()
        return value

    def autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype != torch.float32,
        )

    def read_audio(self, path):
        from .data import load_mono_audio

        with self.timer.measure("audio_decode_resample_cpu"):
            return load_mono_audio(Path(path), self.processor.audio_sampling_rate)

    @torch.no_grad()
    def mixture(self, wave):
        with self.timer.measure("audio_processor_cpu"):
            batch = self.processor(descriptions=[""], audios=[wave])
        with self.timer.measure("audio_transfer_to_gpu"):
            batch = batch.to(self.device)
        with self.autocast(), self.timer.measure("dac_mixture_encode"):
            audio = self.sam._get_audio_features(batch.audios)
        return self.finish(
            {
                "forward_args": {
                    "audio_features": audio,
                    "audio_pad_mask": batch.audio_pad_mask,
                    "anchor_ids": batch.anchor_ids,
                    "anchor_alignment": batch.anchor_alignment,
                },
                "num_samples": wave.shape[-1],
                "latent_frames": audio.shape[1],
            }
        )

    @torch.no_grad()
    def video(self, path, mixture):
        from sam_audio.processor import load_video

        with self.timer.measure("video_decode_select_cpu"):
            frames = load_video(
                torch.tensor([mixture["latent_frames"]]),
                [path],
                self.processor.feature_to_wav_idx,
                self.processor.audio_sampling_rate,
            )
        with self.autocast(), self.timer.measure("pe_resize_and_encode"):
            if self.video_frame_chunk > 0:
                if len(frames) != 1:
                    raise ValueError("Cached extraction expects one video per call")
                video = frames[0]
                encoded = [
                    self.sam.vision_encoder(
                        [video[start : start + self.video_frame_chunk].to(self.device)]
                    )
                    for start in range(0, video.size(0), self.video_frame_chunk)
                ]
                features = torch.cat(encoded, dim=1).transpose(1, 2)
            else:
                with self.timer.measure("video_transfer_to_gpu"):
                    frames = [v.to(self.device) for v in frames]
                features = self.sam.vision_encoder(frames).transpose(1, 2)
        return self.finish(features)

    @torch.no_grad()
    def text(self, prompt):
        with self.autocast(), self.timer.measure("t5_tokenize_and_encode"):
            features, mask = self.sam.text_encoder([prompt])
        return self.finish({"text_features": features, "text_mask": mask})

    @torch.no_grad()
    def clean(self, target, residual=None):
        if getattr(self.sam,"_target_only_conversion",None):
            if residual is not None:raise RuntimeError("Target-only path received a residual")
            with self.autocast(), self.timer.measure("dac_target_encode"):
                z = self.sam.audio_codec(target.to(self.device).unsqueeze(0)).transpose(1,2)
            if z.shape[-1]!=128:raise RuntimeError("Target codec channel mismatch")
            return self.finish(z)
        with self.timer.measure("target_residual_transfer_to_gpu"):
            target, residual = target.to(self.device), residual.to(self.device)
        with self.autocast(), self.timer.measure("dac_target_encode"):
            target_latent = self.sam.audio_codec(target.unsqueeze(0)).transpose(1, 2)
        with self.autocast(), self.timer.measure("dac_residual_encode"):
            residual_latent = self.sam.audio_codec(residual.unsqueeze(0)).transpose(
                1, 2
            )
        return self.finish(torch.cat((target_latent, residual_latent), dim=-1))

    def sample(self, sample):
        mixture = self.mixture(sample["mixture"])
        args = dict(mixture["forward_args"])
        args["masked_video_features"] = self.video(sample["video_path"], mixture)
        args.update(self.text(sample["prompt"]))
        return {
            "forward_args": args,
            "clean": self.clean(sample["target"], sample.get("residual")),
        }


def build_components(plan, store, extractor, *, rank=0, world_size=1):
    """Shard mixture groups (including targets) and distinct texts across GPUs.

    Needs shared local storage when world_size > 1. There is one writer per key;
    caller places a barrier before finalizing/reading the catalog.
    """
    from .data import _fit_to_length

    started = time.perf_counter()
    for key, spec in sorted(plan["mixtures"].items()):
        if int(key, 16) % world_size != rank:
            continue
        need_mix = not store.exists("mixture", key)
        need_clean = any(not store.exists("clean", k) for k in spec["targets"])
        wave = None
        if need_mix or need_clean:
            wave = extractor.read_audio(spec["path"])
            if spec["max_samples"] is not None:
                wave = wave[..., : spec["max_samples"]]
        if need_mix:
            with extractor.timer.measure("cache_mixture_write_inclusive"):
                store.put("mixture", key, extractor.mixture(wave))
        mixture = store.get("mixture", key)
        for vkey, path in spec["videos"].items():
            if not store.exists("video", vkey):
                store.put("video", vkey, extractor.video(path, mixture))
        for ckey, target_spec in spec["targets"].items():
            if store.exists("clean", ckey):
                continue
            length = mixture["num_samples"]
            target = _fit_to_length(extractor.read_audio(target_spec["target"]), length)
            if target_spec.get("target_only",False):
                store.put("clean", ckey, extractor.clean(target))
                continue
            residual = (
                _fit_to_length(extractor.read_audio(target_spec["residual"]), length)
                if target_spec["residual"]
                else wave - target
            )
            store.put("clean", ckey, extractor.clean(target, residual))
    for key, prompt in sorted(plan["texts"].items()):
        if int(key, 16) % world_size == rank and not store.exists("text", key):
            store.put("text", key, extractor.text(prompt))
    extractor.timer.sync()
    return {
        "rank": rank,
        "seconds": time.perf_counter() - started,
        "new_components": store.counts,
        "diagnostic_stages": extractor.timer.report(),
    }


def finalize_catalog(plan, store, sample_rate):
    for split in plan["splits"].values():
        for entry in split["entries"]:
            refs = entry["variants"][0]
            mix = store.get("mixture", refs["mixture"])
            entry["latent_frames"] = mix["latent_frames"]
            entry["duration"] = mix["num_samples"] / sample_rate
            for variant in entry["variants"]:
                for kind in ("mixture", "video", "clean", "text"):
                    if not store.exists(kind, variant[kind]):
                        raise FileNotFoundError(store.path(kind, variant[kind]))
    catalog = {
        "plan_id": plan["id"],
        "signature": store.signature,
        "splits": plan["splits"],
    }
    path = store.root / f"catalog-{plan['id']}.json"
    atomic_json(path, catalog)
    return path


class CachedTrainingDataset(Dataset):
    def __init__(self, catalog, split, store, *, online_dataset=None, referring_pairs=None):
        self.entries = catalog["splits"][split]["entries"]
        self.seed = catalog["splits"][split]["seed"]
        self.store, self.online_dataset = store, online_dataset
        self.referring_pairs = referring_pairs
        if referring_pairs is not None and online_dataset is not None:
            raise ValueError("Referring auxiliary requires cached features")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, slot):
        epoch, index, valid, denominator, last_micro, update = slot
        entry = self.entries[index]
        variant = select_variant(self.seed, epoch, index, len(entry["variants"]))
        start = time.perf_counter()
        if self.online_dataset is None:
            value = self.store.assemble(entry["variants"][variant])
            if self.referring_pairs is not None:
                value = self.referring_pairs.attach(value, entry["sample_id"], entry["variants"][variant], self.store)
        else:
            self.online_dataset.set_epoch(epoch)
            value = self.online_dataset[index]
        return {
            "value": value,
            "valid": valid,
            "denominator": denominator,
            "last_micro": last_micro,
            "update": update,
            "duration": entry["duration"],
            "sample_id": entry["sample_id"],
            "worker_read_seconds": time.perf_counter() - start,
            "cached": self.online_dataset is None,
        }


def collate_cached_training(items):
    for key in ("denominator", "last_micro", "update", "cached"):
        if len({item[key] for item in items}) != 1:
            raise ValueError(f"Inconsistent batch {key}")
    cached = items[0]["cached"]
    # collate_features creates a correct right-zero-padded batch and preserves
    # audio_pad_mask.  The sampler decides whether lengths must be exact.  Keep
    # this collator capable of variable lengths so the empirically validated
    # padded-bucket ablation can use the same loss and data path.
    return {
        "features": collate_features([v["value"] for v in items]) if cached else None,
        "samples": None if cached else [v["value"] for v in items],
        "valid": torch.tensor([v["valid"] for v in items], dtype=torch.float32),
        "duration": torch.tensor([v["duration"] for v in items], dtype=torch.float64),
        "worker_read_seconds": sum(v["worker_read_seconds"] for v in items),
        "sample_id": [v["sample_id"] for v in items],
        **{k: items[0][k] for k in ("denominator", "last_micro", "update", "cached")},
    }
