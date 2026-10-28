# Qwen30B + SAM Audio joint training

This branch contains the **V6 source to run and edit**, directly in Git. No source ZIP download is needed.

**Start reading:** [training entry](src/portable_train.py) · [training updates](src/distributed_trainer.py) · [model construction](src/production_models.py) · [Slurm script](scripts/train.sbatch).

One Linux x86_64 node, four ~48 GB NVIDIA GPUs, 16 CPUs, 128 GiB host RAM. Per-rank batch 2, global batch 8. The supplied launcher uses Slurm. The joint Qwen / causal AUDSEG bridge / SAM training implementation and V6 acceleration modules are retained.

## 1. Get the code

```bash
git clone https://github.com/TerryShenty/sam-audio-workspace.git
cd sam-audio-workspace
```

GitHub **Code → Download ZIP** also works: extract the repository and open its directory. All commands below run from this directory. Choose a writable shared storage directory; replace `/YOUR_STORAGE/qwen-sam` throughout. Keep this source checkout available to the compute node.

## 2. Set up an environment

On CPU/transfer facilities with network access, Python 3.12, git and a C++ compiler:

```bash
python3.12 qwen_quickstart.py setup --root /YOUR_STORAGE/qwen-sam
```

This installs a private environment; it does **not** submit a GPU job or download another source package. The reference installer selects CUDA 13, so check your NVIDIA driver/site modules. For your own environment instead, see [environment options](docs/ENVIRONMENT.md). You may adapt dependencies for your server; a different stack is not yet GPU-validated.

## 3. Sign in and prepare data

Request access to [facebook/sam-audio-large-tv](https://huggingface.co/facebook/sam-audio-large-tv), then log in with your own account:

```bash
/YOUR_STORAGE/qwen-sam/environment/bin/hf auth login
python3.12 qwen_quickstart.py prepare --root /YOUR_STORAGE/qwen-sam
```

Preparation downloads official weights and the [public joint dataset](https://huggingface.co/datasets/TerryShenty/qwen30b-sam-joint-111k), checks identities, builds CPU extensions and prepares the frozen audio codec. Dataset: 111,068 training + 817 validation rows, about 350.7 GB of tar payload; allow extra space for models, extraction and checkpoints. Run this on CPU/transfer facilities, not a GPU allocation.

Already have the models and data? Copy `configs/paths.example.json` to your own paths file, fill in the four paths, and run `python scripts/prepare.py --paths /absolute/path/to/paths.json` using your chosen environment. Existing deployment bindings must match the current checkout; keep previous run data/bindings intact when relocating code.

## 4. Start training directly

```bash
python3.12 qwen_quickstart.py train --root /YOUR_STORAGE/qwen-sam \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

Replace account/partition with your cluster values. This submits **one 24-hour job**, with one torchrun and four persistent ranks. **No separate qualification job is required.** `qualify` is optional debugging only. Three startup warmup updates are real updates and are retained. There is no epoch cap or automatic renewal. Edit site resource directives in [scripts/train.sbatch](scripts/train.sbatch) before submission as needed; preserve four GPUs/four ranks/global 8 for this implementation.

The call chain is `qwen_quickstart.py → scripts/train.sbatch → scripts/launch.py → src/portable_train.py`. Do not run `portable_train.py` alone: the launcher supplies the distributed environment and bound run plan.

## 5. Results

Default output: `/YOUR_STORAGE/qwen-sam/runs/train/`.

- `training-rank0.csv` … `training-rank3.csv`: per-rank training records.
- `validation/step-*-summary.csv` and `step-*-samples.csv`: full 817-example validation every half epoch.
- `VALIDATION817-summary.csv`, `VALIDATION817-samples.csv`: final validation.
- `checkpoints/step-*/`: complete resumable training state. Keep this directory; CSVs are not checkpoints.

Training saves a full checkpoint before each half-epoch validation, then continues. The fixed wall limit takes priority; the run reserves about 30 minutes for final validation. A failed validation does not count as completion. See [method and validation](docs/METHOD_AND_VALIDATION.md) for the protocol and [development](docs/DEVELOPMENT.md) for editing/resume.

## Code map

| Location | Purpose |
|---|---|
| `qwen_quickstart.py` | Setup, preparation and submission convenience entry |
| `scripts/` | Environment/data helpers, preparation and Slurm launcher |
| `configs/` | Effective V6 settings, SAM settings, path template and model identities |
| `src/portable_train.py` | Main distributed training lifecycle |
| `src/distributed_trainer.py` | Actual training update |
| `src/production_models.py` | Model and optimizer construction |
| `src/production_loop.py` | Training/checkpoint/validation scheduling |
| `src/upstream_v2/` | Required bundled kernels/runtime dependencies; retain these |
| `docs/` | Environment, development and method details |

## Versions and verification

`main` is the current collaborator source. The previous ZIP-based handoff and other historical workspace files are preserved on [archive/pre-source-layout-20261028](https://github.com/TerryShenty/sam-audio-workspace/tree/archive/pre-source-layout-20261028). Existing historical files can also be read at commit `93eeca21e31da1ccaae5fde14c13425a7cefe74e`.

This reorganization preserves the V6 numerical training implementation. Source/layout and mocked submission checks are not a new-server installation test, a four-GPU training pass, or full 817 GPU validation. The existing single-A100/global-2 speed evidence is not a four-GPU speed guarantee.
