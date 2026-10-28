"""Dataset adapter for the benchmark's ``training_metadata/prompts.csv``."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _approved(value: str | None) -> bool:
    # Existing training metadata uses blank to mean not rejected.
    if value is None or not value.strip():
        return True
    return value.strip().lower() in {"1", "true", "yes", "y", "approved"}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mono_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """Decode WAV or the audio track of a media container as ``[1, N]``."""

    try:
        waveform, source_rate = torchaudio.load(str(path))
    except Exception:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path), sample_rate=sample_rate, num_channels=1)
        waveform = decoder.get_all_samples().data
        source_rate = sample_rate
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if waveform.numel() == 0:
        raise RuntimeError(f"Decoded empty audio: {path}")
    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"Decoded non-finite audio: {path}")
    return waveform


def _fit_to_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.size(-1) >= length:
        return waveform[..., :length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.size(-1)))


def _stable_split_groups(
    group_ids: Iterable[str],
    validation_fraction: float,
    split_seed: int,
) -> set[str]:
    unique = sorted(set(group_ids))
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0:
        return set()
    # Never move the only source group in a stratum entirely to validation.
    # This matters for an early dataset version with a very small domain.
    if len(unique) < 2:
        return set()
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    scored = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"{split_seed}:{value}".encode("utf-8")
        ).digest(),
    )
    return set(scored[:count])


class NativeReferringDataset(Dataset):
    """One item per target; one of its paraphrases is sampled each epoch.

    Multiple targets in one video are deliberately kept as separate supervised
    examples.  The target stem determines ``x_tgt`` and ``x_res = x_mix-x_tgt``;
    no wrong-target contrastive term is needed for this native fine-tuning stage.
    """

    required_columns = {
        "clip_id",
        "video_path",
        "target_source_id",
        "gt_audio_path",
        "prompt",
    }

    def __init__(
        self,
        manifest: str | Path,
        *,
        data_root: str | Path,
        sample_rate: int = 48_000,
        max_seconds: float | None = 10.0,
        split: str = "train",
        validation_fraction: float = 0.1,
        split_seed: int = 17,
        split_group_column: str = "video_id",
        split_stratify_column: str | None = "domain",
        seed: int = 0,
        max_targets: int | None = None,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_samples = (
            None if max_seconds is None else round(sample_rate * max_seconds)
        )
        self.seed = seed
        self.epoch = 0

        rows = [row for row in _read_csv(self.manifest) if _approved(row.get("approved"))]
        if not rows:
            raise ValueError(f"No enabled rows in {self.manifest}")
        missing = self.required_columns.difference(rows[0])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        def split_group_id(row: dict[str, str]) -> str:
            # Prefix with dataset so coincidentally reused video IDs in two
            # corpora do not become one source group.
            group = row.get(split_group_column) or row["clip_id"]
            dataset = row.get("dataset", "").strip()
            return f"{dataset}::{group}" if dataset else group

        groups_by_stratum: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            stratum = (
                row.get(split_stratify_column, "unknown")
                if split_stratify_column
                else "all"
            )
            groups_by_stratum[stratum].add(split_group_id(row))
        validation_groups: set[str] = set()
        for stratum, group_values in sorted(groups_by_stratum.items()):
            stratum_seed = int.from_bytes(
                hashlib.sha256(stratum.encode("utf-8")).digest()[:4], "big"
            )
            validation_groups.update(
                _stable_split_groups(
                    group_values,
                    validation_fraction,
                    split_seed + stratum_seed,
                )
            )
        if split != "all":
            want_validation = split == "validation"
            rows = [
                row
                for row in rows
                if (split_group_id(row) in validation_groups)
                == want_validation
            ]

        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[
                (row.get("dataset", ""), row["clip_id"], row["target_source_id"])
            ].append(row)
        keys = sorted(grouped)
        if max_targets is not None:
            rng = random.Random(split_seed + (0 if split == "train" else 1))
            rng.shuffle(keys)
            keys = sorted(keys[:max_targets])
        self.examples = [grouped[key] for key in keys]
        if not self.examples:
            raise ValueError(f"The {split} split contains no targets")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def domains(self) -> list[str]:
        return [rows[0].get("domain", "unknown") for rows in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.examples[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        row = rows[rng.randrange(len(rows))]

        if getattr(self, "require_explicit_mixture_path", False) and not row.get("mixture_audio_path", "").strip():
            raise ValueError("This training recipe requires an explicit mixture_audio_path, including when it equals video_path")

        video_path = _resolve(self.data_root, row["video_path"])
        mixture_path = _resolve(
            self.data_root, row.get("mixture_audio_path") or row["video_path"]
        )
        target_path = _resolve(self.data_root, row["gt_audio_path"])
        mixture = load_mono_audio(mixture_path, self.sample_rate)
        target = load_mono_audio(target_path, self.sample_rate)

        length = mixture.size(-1)
        if self.max_samples is not None:
            length = min(length, self.max_samples)
        mixture = mixture[..., :length]
        target = _fit_to_length(target, length)
        if getattr(self, "target_only", False):
            residual = None  # No residual path read, subtraction, or encoding.
        elif row.get("residual_audio_path"):
            residual = _fit_to_length(
                load_mono_audio(
                    _resolve(self.data_root, row["residual_audio_path"]),
                    self.sample_rate,
                ),
                length,
            )
        else:
            # This exact identity is important because SAM Audio jointly learns
            # target and everything-else residual channels.
            residual = mixture - target

        return {
            "sample_id": (
                f"{row.get('dataset', '')}::{row['clip_id']}::"
                f"{row['target_source_id']}"
            ),
            "clip_id": row["clip_id"],
            "dataset": row.get("dataset", ""),
            "video_id": row.get("video_id") or row["clip_id"],
            "target_source_id": row["target_source_id"],
            "domain": row.get("domain", "unknown"),
            "prompt": row["prompt"].strip(),
            "video_path": str(video_path),
            "mixture": mixture,
            "target": target,
            **({"residual": residual} if residual is not None else {}),
            "num_prompt_variants": len(rows),
        }


def collate_native_referring(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = torch.tensor([item["mixture"].size(-1) for item in samples])

    def pad(key: str) -> torch.Tensor:
        return pad_sequence(
            [item[key].transpose(0, 1) for item in samples],
            batch_first=True,
        ).transpose(1, 2)

    return {
        "sample_id": [item["sample_id"] for item in samples],
        "clip_id": [item["clip_id"] for item in samples],
        "dataset": [item.get("dataset", "") for item in samples],
        "video_id": [item["video_id"] for item in samples],
        "target_source_id": [item["target_source_id"] for item in samples],
        "domain": [item["domain"] for item in samples],
        "prompt": [item["prompt"] for item in samples],
        "video_path": [item["video_path"] for item in samples],
        "mixture": pad("mixture"),
        "target": pad("target"),
        **({"residual": pad("residual")} if "residual" in samples[0] else {}),
        "audio_sizes": sizes,
    }
