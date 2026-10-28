# Qwen30B + SAM Audio joint training

This branch contains the **V6 source to run and edit**, directly in Git. No source ZIP download is needed.

**Start reading:** [training entry](src/portable_train.py) · [training updates](src/distributed_trainer.py) · [model construction](src/production_models.py) · [Slurm script](scripts/train.sbatch).

One Linux x86_64 node, 1–8 ~48 GB NVIDIA GPUs, 4 CPUs and 32 GiB host RAM requested per GPU by default. Per-rank batch stays 2; global batch is 2 × GPU count. The supplied launcher uses Slurm. The joint Qwen / causal AUDSEG bridge / SAM training implementation and V6 acceleration modules are retained.

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
  --account YOUR_ACCOUNT --partition YOUR_PARTITION --gpus 4
```

Replace account/partition with your cluster values. This submits **one 24-hour job**, with one torchrun and the requested number of persistent ranks. **No separate qualification job is required.** `qualify` is optional debugging only. Three startup warmup updates are real updates and are retained. There is no epoch cap or automatic renewal. Edit site resource directives in [scripts/train.sbatch](scripts/train.sbatch) before submission as needed; keep the resource request and `--gpus` consistent.

The call chain is `qwen_quickstart.py → scripts/train.sbatch → scripts/launch.py → src/portable_train.py`. Do not run `portable_train.py` alone: the launcher supplies the distributed environment and bound run plan.

### Choose your GPU count

Use `--gpus 1`, `--gpus 2`, `--gpus 4`, or `--gpus 8` (any integer 1–8, one node). Enter the number allocated to this job, not all GPUs owned by your lab. The Slurm request, torchrun processes, sampler, validation and checkpoints follow that number. GPUs become visible after Slurm allocates them; the worker checks the count before loading models. The launcher never takes every idle GPU automatically.

Per-device batch remains **2** to preserve the accepted V6 fast path. Global batch is therefore 2/4/8/16 for 1/2/4/8 GPUs; the launch command prints it. This is not fixed-global-batch equivalence, and there is no new gradient accumulation policy or automatic learning-rate scaling. V6 kernels/configuration remain unchanged. One GPU bypasses DDP wrapping; multiple GPUs retain DDP averaging for all trainable groups. Each GPU holds a model replica; more GPUs do not reduce per-device model memory.

Complete-state resume requires the **same GPU count and configuration**. Changing GPU count while resuming is rejected; resharding saved RNG/sampler/optimizer state is not implemented. The CPU/Gloo integration test covers real tiny-model updates and resume at 1/2/4/8 ranks; target CUDA/NCCL and full Qwen+SAM runs at these counts still require measurement.

## 5. Results

Default output: `/YOUR_STORAGE/qwen-sam/runs/train/`.

- `training-rank0.csv` … `training-rank<N-1>.csv`: per-rank training records.
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
