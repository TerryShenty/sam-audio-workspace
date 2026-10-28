# Editing and resuming

Read `src/portable_train.py`, then `src/production_build.py`, `src/distributed_trainer.py` and `src/production_models.py`. Optimization modules under `src/upstream_v2/` are dependencies of V6, not alternative training versions to select manually.

## Local changes

Use an isolated checkout for edits. Do not change files in a checkout referenced by a pending or running job. Source manifests record exactly what a run executes. After reviewing your changes, run:

```bash
python3.12 scripts/update_manifest.py
python3.12 scripts/verify_package.py
```

The first command explicitly records the new source hashes; it does not approve correctness or GPU readiness. Re-run CPU preparation with your environment after affected changes. Internal baseline/kernel identity checks remain active; changing an identity-bound kernel requires reviewing its affected contract, not blindly bypassing a check. Keep changed-source experiments distinct from the original V6 evidence. No need to reseal for untracked local paths or generated run files.

## Existing environment

Pass `--python /absolute/path/to/environment/bin/python` to setup, prepare and train. Setup then checks that environment's imports and audio decoding without installing packages. Ensure FFmpeg shared libraries are visible in `LD_LIBRARY_PATH` where your site requires it. Preparation and training use the same selected Python. `requirements.txt` is the direct runtime list; see ENVIRONMENT.md and the optional installer for the full reference stack/upstream dependencies.

## Complete-state resume

Use a new output directory, the same compatible code/config/data, and the complete checkpoint directory (with COMPLETE.json and all rank state). Do not substitute a research/qualification checkpoint for production initialization.

```bash
export QWEN_PACKAGE="$(pwd)"
export QWEN_PYTHON=/YOUR_STORAGE/qwen-sam/environment/bin/python
export QWEN_OUTPUT=/YOUR_STORAGE/qwen-sam/runs/resumed
export QWEN_MODE=train
export QWEN_GPUS=4  # must match the checkpoint and sbatch GPU request
export QWEN_RESUME=/absolute/path/to/previous/checkpoints/step-XXXXXXXXXXXX
unset QWEN_GATE
sbatch --gres=gpu:4 --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/train.sbatch
```

This explicitly requests a new allocation; inspect existing jobs and submission receipts first. The convenience `train` entry intentionally does not inherit an old QWEN_RESUME value. No automatic retries or renewal are performed.

## GPU-count integration checks

`python tests/test_world_size.py` exercises real CPU/Gloo DDP at 1/2/4/8 ranks, agreement with a global-batch reference, complete-state restoration and effective continuation, sampler/817 coverage, and rejection of cross-world resume. It does not execute the CUDA kernels or real 30B model. Per-rank batch stays 2. Do not change GPU count on checkpoint resume.
