# Direct four-GPU, 24-hour Qwen30B + SAM Audio training

Use [QWEN_QUICKSTART.md](QWEN_QUICKSTART.md) and `qwen_quickstart.py`: setup, HF login, prepare, then train. No separate qualification job or gate receipt is required. The optional qualify action is for debugging only.

The V6 joint model, optimizer, loss, caches, four persistent ranks and global batch 8 are unchanged. Full 817-example validation runs every half epoch (0.5, 1, 1.5, 2, 2.5, ...), after a complete checkpoint, and training continues afterward. The original walltime and checkpoint/final-validation reserves still apply. CSV results are under `<root>/runs/train/`; per-boundary results are under its `validation/` directory.

This change removes the separate short-test prerequisite; it does not establish target GPU success. Single-A100 fixed64/global2 mean 0.809298911 seconds is not a four-GPU speed guarantee. New-machine installation and full four-GPU execution remain unverified. Existing source/data guards, finite training checks, watchdogs, full-state checkpoints and validation memory probes are retained. No existing Longleaf job is changed.
