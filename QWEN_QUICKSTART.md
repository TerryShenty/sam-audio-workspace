# Qwen30B + SAM Audio: simplified entry

Use this entry for Qwen joint training. The repository-root `quickstart.sh` is a separate SAM-only workflow.

Prerequisites: activate a compatible Linux Python 3.12 environment with the dependencies in [the V6 guide](QWEN30B_V6_README.md); sign in using `hf auth login` and obtain official SAM model access. The environment is not automatically installed or validated by this wrapper. Use CPU/data-transfer facilities with a compiler for preparation, not GPUs. Allow at least 350.57 GB for data plus models, checkpoints and temporary space.

Download the entry once:

```bash
curl -fL https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/qwen_quickstart.py -o qwen_quickstart.py
```

One preparation command downloads the pinned, hash-checked V6 source archive, official model weights and completed dataset, fills paths automatically, then runs CPU preparation:

```bash
python qwen_quickstart.py prepare --root /YOUR_STORAGE/qwen-sam
```

The command refuses to proceed while the HF release lacks `UPLOAD_COMPLETE.json`. Official model access errors require your own HF account/access. Source/model identity checks are not bypassed. Re-running preparation reuses the existing download caches.

Submit one four-GPU qualification job (adjust the bundled Slurm resource template to your site if necessary):

```bash
python qwen_quickstart.py qualify --root /YOUR_STORAGE/qwen-sam --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

After qualification succeeds, submit training:

```bash
python qwen_quickstart.py train --root /YOUR_STORAGE/qwen-sam --account YOUR_ACCOUNT --partition YOUR_PARTITION --gate /YOUR_STORAGE/qwen-sam/runs/qualify/PORTABLE_GATE.json
```

Submission is journaled before `sbatch`; a repeated request with the same output is blocked. On an interrupted or failed submission, inspect the journal and scheduler before deciding to retry. Use a new `--output` only for an intentional new run. No automatic resubmission or renewal. Formal checkpoint resume uses the detailed V6 guide, not this simplified fresh-run entry.

This wrapper merges existing operations; it does not establish fresh-environment GPU acceptance. Full four-GPU training/resume and the complete 817-example validation remain unverified. The validation API still needs codec/memory integration; training does not automatically run full validation. See [method and validation](QWEN30B_METHOD_AND_VALIDATION.md).
