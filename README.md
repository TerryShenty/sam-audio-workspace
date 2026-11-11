# Qwen30B + SAM Audio: four-arm objective ablation on animal-3000 / General Sound

Same model, data, sampler and optimizer in every arm; only the training objective differs.

| arm | objective (per example, on top of the unchanged V6 joint path) |
|---|---|
| `baseline` | 0.5·FM_target + 0.5·FM_residual + 0.1·Qwen CE, flow time t ~ U(0,1) |
| `dual_t0` | FM terms = ½·FM(t ~ U(0,1)) + ½·FM(t = 0) for the same example (A and B share the t = 0 noise) |
| `contrastive` | baseline + λ·D, D = ‖(v_A − v_B) − (c_A − c_B)‖² at t = 0 with the **same** noise for A and B |
| `dual_t0_contrastive` | dual_t0 FM average + λ·D (one shared t = 0 forward) |

The arms form a 2×2 (FM time points × D), so `dual_t0_contrastive` differs from `dual_t0` only by D and `contrastive` differs from `baseline` only by D; if GPUs are limited, `baseline` → `dual_t0` → `dual_t0_contrastive` still isolates each addition, and `contrastive` can be run later. D is computed only at t = 0. t = 0 is pure noise in SAM Audio's convention, so at t = 0 the A and B inputs are identical and only the prompt can make the two predictions differ. λ = 1. Implementation: [`src/ablation_joint.py`](src/ablation_joint.py).

**Model.** Qwen3-Omni-30B-A3B-Thinking (NF4, q/v LoRA r8) → causal `<AUDSEG>` hidden state → bridge → one SAM conditioning token; SAM Audio Large-TV with the seven trained groups (attention LoRA, FFN LoRA, input projection, output head, context embedder, memory projection, video aligner). Fresh official initialization. The V6 accelerated runtime is reused unchanged ([V6 notes](docs/V6_JOINT_TRAINING_README.md), [method](docs/METHOD_AND_VALIDATION.md)).

**Data.**
- Training: 3,000 synthetic two-animal videos, 6,000 referring targets (A and B of each video; mixture = A + B). Every update puts one mixture's A and B on each GPU ([`src/ablation_sampler.py`](src/ablation_sampler.py)); per-GPU batch 2, global batch 2 × GPUs; 30 epochs.
- Validation, after every epoch, **reported separately, never pooled**:
  - `check`: 200 training videos / 400 targets that are also trained on (can the model learn its training set?). Checkpoint selection uses this set.
  - `gs_benchmark`: all General Sound prompts of the referring benchmark (181 videos / 323 targets / 413 prompts at commit `54c3fda5`), real videos, report only.

**Speed choices.** Training keeps the accepted V6 fast path (compiled SAM region, NF4/INT8 Qwen kernels) unchanged. In the three non-baseline arms the random-t and t = 0 SAM passes run as two batch-2 calls that reuse one Qwen condition: the V6 compiled SAM region covers batch 1–2 only, so a merged batch-4 call would be slower. Validation runs 8 rows per GPU by default (`ABL_VALIDATION_BATCH`), longest rows first; on CUDA OOM a batch is split in half and retried down to 1, no row is dropped, and splits are recorded in `oom_batch_splits`. Batches above 2, and rows that are not 250/256/500 latent frames (most GS clips), use the uncompiled SAM path; per-row noise seeds keep results independent of batching up to numerical differences.

**Scoring** ([`src/referring_metrics_v2.py`](src/referring_metrics_v2.py)): aligned SI-SDR (±50 ms), no epsilon in the ratio and no clipping of per-row values; an all-zero prediction is undefined (`pred_silent`, a failure) instead of 0 dB; quiet but non-zero outputs are scored normally and only flagged; a silent reference is excluded and counted; if the reference equals the mixture, SI-SDRi is undefined. Summaries: mean with values below −30 dB counted as −30 dB (**selection metric**, on `check`), share below −30 dB, finite mean, median/IQR, SI-SDRi failure rates, clip-bootstrap CI.

## Steps

All commands run from the repository root. Choose a shared storage root `R`.

**1. Environment (CPU/network)** — unchanged V6 installer:
```bash
python3.12 qwen_quickstart.py setup --root R
R/environment/bin/hf auth login      # your own HF account
```

**2. Models and data (transfer/CPU)**
```bash
R/environment/bin/python scripts/download_models.py --model-root R/models
R/environment/bin/python scripts/download_ablation_data.py --destination R/raw
```
The benchmark is a private dataset: your HF account needs read access. Only its metadata and General Sound files are downloaded; do not redistribute them.

**3. Data contract and CPU preparation**
```bash
R/environment/bin/python scripts/ablation_prepare.py --animal R/raw/animal3000 --benchmark R/raw/benchmark --out R/data
R/environment/bin/python scripts/ablation_cpu_prepare.py --data-root R/data --model-root R/models
```

**4. Caches (GPU; you choose the GPU count)** — SAM features, then Qwen processor + frozen audio/video encoder outputs. Each runs one shard per GPU and can be rerun (finished entries are reused).
```bash
sbatch --gres=gpu:N --cpus-per-task=$((4*N)) --mem=$((32*N))G --time=08:00:00 --account=ACCOUNT --partition=PARTITION \
  --export=ALL,STAGE=sam,ABL_PACKAGE=$PWD,ABL_PYTHON=R/environment/bin/python scripts/ablation_cache.sbatch
sbatch ... --export=ALL,STAGE=qwen,ABL_PACKAGE=$PWD,ABL_PYTHON=R/environment/bin/python scripts/ablation_cache.sbatch
```

**5. Train one arm per job** (1–8 GPUs on one node; default 4):
```bash
sbatch --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=24:00:00 --account=ACCOUNT --partition=PARTITION \
  --export=ALL,ABL_ARM=baseline,ABL_OUTPUT=R/runs/baseline-1,ABL_PACKAGE=$PWD,ABL_PYTHON=R/environment/bin/python \
  scripts/ablation_train.sbatch
```
Repeat for `dual_t0`, `contrastive`, `dual_t0_contrastive` with their own output directories. A job stops at a checkpoint before its time limit if 30 epochs are not finished; continue in a new job with a new `ABL_OUTPUT` and `ABL_RESUME=<old output>/checkpoints/step-…` (same arm and GPU count; resharding is not supported). Set `ABL_GPUS` to match `--gres`.

## Outputs (per run directory)
- `validation/epoch-NNN/summary.json` — both sets' summaries (`sets.check`, `sets.gs_benchmark`), `selection_metric`, `headline`; `*_rows.jsonl` per-row scores.
- `epochs.jsonl` — one line per epoch: checkpoint path, selection value, headline numbers of both sets.
- `terms.jsonl` — every 20 steps: per-example target/residual FM, CE, accuracy, and for non-baseline arms the random-t and t = 0 FM terms and the pair-difference term.
- `checkpoints/step-…` — complete state (model, optimizer, scheduler, EMA, all-rank RNG, sampler position, configuration); kept, never deleted automatically.

## What has and has not been checked
- CPU tests (`python -m pytest -q tests/test_ablation_cpu.py`, 17 tests): validation OOM batch splitting, pair sampler coverage/resume for 1–8 ranks, each arm's loss against its definition on a stand-in SAM (identical t = 0 inputs for A/B, gradients reach all parameters), scorer floor/zero-prediction/no-clip/undefined SI-SDRi, data-contract builder, fail-closed loader, epoch-budget loop.
- **Not yet run on GPU**: cache builders, full 30B training of any arm, validation memory, speed. The two extra SAM forward/backward passes in `dual_t0*`/`contrastive*` raise activation memory; run a short job first and check per-GPU memory. V6 timing numbers do not transfer to these arms.

## Code map
| file | role |
|---|---|
| `scripts/ablation_prepare.py` | raw data → `MEDIA.json`, `ROWS.json`, SAM cache plan (CPU) |
| `scripts/build_sam_cache.py`, `scripts/build_qwen_cache.py` | GPU cache builders (same keys training reads) |
| `scripts/ablation_launch.py` → `src/ablation_train.py` | one torchrun; epoch budget, per-epoch validation, resume |
| `src/ablation_build.py` | V6 model/optimizer/trainer composition with the ablation data/objective |
| `src/ablation_joint.py`, `src/ablation_sampler.py`, `src/ablation_data.py` | objective, A/B pair sampler, data contract |
| `src/ablation_validation.py`, `src/referring_metrics_v2.py` | generation + scoring of both validation sets |

Based on [`TerryShenty/sam-audio-workspace`](https://github.com/TerryShenty/sam-audio-workspace) commit `83e329a` (V6); its history is kept in this repository.
