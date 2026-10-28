"""CSV-backed data loading for same-video hard-negative training."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset


def _is_enabled(value: str | None) -> bool:
    if value is None or value == "":
        return True
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _fit_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.size(-1) >= length:
        return waveform[..., :length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.size(-1)))


def load_mono_audio(path: str | Path, sample_rate: int, length: int) -> torch.Tensor:
    """Load one source as ``[1, samples]`` and pad/crop deterministically."""

    path = str(path)
    try:
        waveform, source_rate = torchaudio.load(path)
    except Exception:
        # torchcodec handles media containers such as MP4 more reliably.
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(path, sample_rate=sample_rate, num_channels=1)
        waveform = decoder.get_all_samples().data
        source_rate = sample_rate
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return _fit_length(waveform, length)


class ReferringAudioDataset(Dataset):
    """One row per referring prompt, grouped with every source in its video.

    Required prompt-manifest columns:
      ``clip_id, video_path, prompt, target_source_id, gt_audio_path``.

    A separate target manifest is optional.  If supplied, it must contain
    ``clip_id, target_source_id, gt_audio_path`` and should list every isolated
    source in the video.  Those rows are used both to construct target/residual
    mixtures and to sample wrong-target hard negatives.
    """

    required_prompt_columns = {
        "clip_id",
        "video_path",
        "prompt",
        "target_source_id",
        "gt_audio_path",
    }
    required_target_columns = {"clip_id", "target_source_id", "gt_audio_path"}

    def __init__(
        self,
        prompt_manifest: str | Path,
        *,
        target_manifest: str | Path | None = None,
        data_root: str | Path | None = None,
        sample_rate: int = 48_000,
        clip_seconds: float = 10.0,
        num_negatives: int = 1,
        seed: int = 0,
        require_negatives: bool = True,
    ):
        self.prompt_manifest = Path(prompt_manifest)
        self.data_root = Path(data_root) if data_root else self.prompt_manifest.parent
        self.sample_rate = sample_rate
        self.num_samples = round(sample_rate * clip_seconds)
        self.num_negatives = num_negatives
        self.seed = seed

        rows = [
            row
            for row in _read_csv(self.prompt_manifest)
            if _is_enabled(row.get("approved"))
        ]
        self._check_columns(rows, self.required_prompt_columns, "prompt manifest")
        self.rows = rows

        target_rows = _read_csv(target_manifest) if target_manifest else rows
        self._check_columns(
            target_rows, self.required_target_columns, "target manifest"
        )
        self.targets_by_clip: dict[str, list[dict[str, str]]] = {}
        seen: set[tuple[str, str, str]] = set()
        for row in target_rows:
            key = (row["clip_id"], row["target_source_id"], row["gt_audio_path"])
            if key in seen:
                continue
            seen.add(key)
            self.targets_by_clip.setdefault(row["clip_id"], []).append(row)

        missing_positive = [
            row
            for row in rows
            if not any(
                target["target_source_id"] == row["target_source_id"]
                for target in self.targets_by_clip.get(row["clip_id"], [])
            )
        ]
        if missing_positive:
            example = missing_positive[0]
            raise ValueError(
                "Target manifest does not contain the requested positive source: "
                f"clip_id={example['clip_id']}, "
                f"target_source_id={example['target_source_id']}"
            )

        if require_negatives and num_negatives > 0:
            invalid = [row["clip_id"] for row in rows if not self._negative_rows(row)]
            if invalid:
                examples = ", ".join(sorted(set(invalid))[:5])
                raise ValueError(
                    "The referring loss needs another source GT in the same clip. "
                    f"No negative was found for clip(s): {examples}"
                )

    @staticmethod
    def _check_columns(
        rows: list[dict[str, str]], required: set[str], label: str
    ) -> None:
        if not rows:
            raise ValueError(f"{label} is empty")
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def _negative_rows(self, prompt_row: dict[str, str]) -> list[dict[str, str]]:
        return [
            row
            for row in self.targets_by_clip.get(prompt_row["clip_id"], [])
            if row["target_source_id"] != prompt_row["target_source_id"]
        ]

    @staticmethod
    def _same_class(a: dict[str, str], b: dict[str, str]) -> bool:
        for key in ("target_class", "instrument_name", "source_class"):
            if a.get(key) and a.get(key) == b.get(key):
                return True
        return False

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sources = self.targets_by_clip[row["clip_id"]]
        positive = next(
            (
                source
                for source in sources
                if source["target_source_id"] == row["target_source_id"]
            ),
            row,
        )
        negatives = self._negative_rows(row)
        negatives.sort(key=lambda item: not self._same_class(positive, item))
        rng = random.Random(self.seed + index)
        same_class_count = sum(self._same_class(positive, item) for item in negatives)
        hard = negatives[:same_class_count]
        easy = negatives[same_class_count:]
        rng.shuffle(hard)
        rng.shuffle(easy)
        ordered_negatives = hard + easy
        if self.num_negatives > len(ordered_negatives) and ordered_negatives:
            ordered_negatives += rng.choices(
                ordered_negatives,
                k=self.num_negatives - len(ordered_negatives),
            )
        selected_negatives = ordered_negatives[: self.num_negatives]

        source_audio = {
            source["target_source_id"]: load_mono_audio(
                self._resolve(source["gt_audio_path"]),
                self.sample_rate,
                self.num_samples,
            )
            for source in sources
        }
        target = source_audio[row["target_source_id"]]

        if row.get("mixture_audio_path"):
            mixture = load_mono_audio(
                self._resolve(row["mixture_audio_path"]),
                self.sample_rate,
                self.num_samples,
            )
        else:
            mixture = torch.stack(list(source_audio.values())).sum(dim=0)
        if row.get("residual_audio_path"):
            residual = load_mono_audio(
                self._resolve(row["residual_audio_path"]),
                self.sample_rate,
                self.num_samples,
            )
        else:
            residual_sources = [
                audio
                for source_id, audio in source_audio.items()
                if source_id != row["target_source_id"]
            ]
            residual = (
                torch.stack(residual_sources).sum(dim=0)
                if residual_sources
                else mixture - target
            )

        # Joint peak scaling preserves mixture = target + residual.
        peak = torch.stack(
            [mixture.abs().max(), target.abs().max(), residual.abs().max()]
        ).max()
        if peak > 0.99:
            scale = 0.99 / peak
            mixture, target, residual = (
                mixture * scale,
                target * scale,
                residual * scale,
            )
            source_audio = {key: value * scale for key, value in source_audio.items()}

        negative_audio = torch.stack(
            [source_audio[item["target_source_id"]] for item in selected_negatives]
        )
        return {
            "sample_id": row.get("sample_id", str(index)),
            "clip_id": row["clip_id"],
            "video_path": str(self._resolve(row["video_path"])),
            "prompt": row["prompt"],
            "target_description": row.get("target_description", "").strip() or None,
            "mixture": mixture,
            "target": target,
            "residual": residual,
            "negatives": negative_audio,
            "negative_source_ids": [
                item["target_source_id"] for item in selected_negatives
            ],
        }


def collate_referring_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-duration samples; every row must have the same negative count."""

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "clip_id": [sample["clip_id"] for sample in samples],
        "video_path": [sample["video_path"] for sample in samples],
        "prompt": [sample["prompt"] for sample in samples],
        "target_description": [sample.get("target_description") for sample in samples],
        "mixture": torch.stack([sample["mixture"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "residual": torch.stack([sample["residual"] for sample in samples]),
        "negatives": torch.stack([sample["negatives"] for sample in samples]),
        "negative_source_ids": [sample["negative_source_ids"] for sample in samples],
    }
