"""Metadata gates before optimized training can consume any target labels."""
import csv
import hashlib
import math
from pathlib import Path


def audit_training_separation(dataset, evaluation_manifest, *, temporal_disjoint_datasets=()):
    path = Path(evaluation_manifest)
    payload = path.read_bytes()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"dataset", "clip_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Protected evaluation manifest needs dataset and clip_id")
        evaluation = list(reader)
    if not evaluation:
        raise ValueError("Protected evaluation manifest is empty")
    def clip(row):
        return row.get("dataset", "").strip(), row["clip_id"].strip()
    def source(row):
        value = row.get("video_id", "").strip()
        return (row.get("dataset", "").strip(), value) if value else None
    eval_clips = {clip(row) for row in evaluation}
    eval_sources = {source(row) for row in evaluation} - {None}
    train_rows = [row for rows in dataset.examples for row in rows]
    temporal = set(temporal_disjoint_datasets)
    by_source = {}
    for row in evaluation:
        by_source.setdefault(source(row), []).append(row)

    def interval(row):
        try:
            start, end = float(row['clip_start_s']), float(row['clip_end_s'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Source-time audit requires explicit clip_start_s/clip_end_s: {clip(row)}") from None
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f"Invalid source-time interval: {clip(row)}")
        return start, end

    overlap, temporal_clips = set(), set()
    for row in train_rows:
        if clip(row) in eval_clips:
            overlap.add(clip(row))
        elif source(row) in eval_sources:
            if row.get('dataset', '').strip() not in temporal:
                overlap.add(clip(row))
                continue
            # This explicit protocol allows disjoint excerpts, not held-out
            # recordings. Do not remove video_id to evade the source check.
            start, end = interval(row)
            for eval_row in by_source[source(row)]:
                eval_start, eval_end = interval(eval_row)
                if max(start, eval_start) < min(end, eval_end):
                    overlap.add(clip(row))
            temporal_clips.add(clip(row))
    overlap = sorted(overlap)
    if overlap:
        raise ValueError(f"Training/evaluation clip or source overlap ({len(overlap)}): {overlap[:8]}")
    return {
        "passed": True, "evaluation_sha256": hashlib.sha256(payload).hexdigest(),
        "training_targets": len(dataset.examples), "evaluation_clips": len(eval_clips),
        "temporal_disjoint_datasets": sorted(temporal),
        "shared_source_disjoint_train_clips": len(temporal_clips),
        "scope": "dataset+clip_id and nonempty dataset+video_id; explicitly configured datasets allow nonoverlapping source-time excerpts, not source-held-out evaluation; cross-dataset shared raw sources require provenance audit",
    }
