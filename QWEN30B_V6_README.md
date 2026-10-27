# Qwen30B + SAM Audio: portable four-GPU V6 handoff

This package contains the selected joint-training runtime, its transitive source dependencies, four native C++ extensions, model download checks, data download/extraction tools, and single-node four-GPU Slurm launchers. It excludes model weights, datasets, credentials, research checkpoints, and automatic speed-comparison sweeps.

**Release status:** source/hash checks and 17 key imports passed in the existing cluster CPU environment (job6923689). This is a debugging handoff, not a recipient-cluster end-to-end acceptance. Four-GPU training/resume and complete817-example validation are still unverified. The training CLI does not yet invoke complete validation automatically.

V6 measured0.809298911 seconds per complete optimizer update on one A100/global batch2/fixed64 inputs, averaged over two matched32-update windows. It was1.64726% faster than the matched control. This is not a four-GPU speed guarantee. V6 adds approximate INT8 attention input gradients; held-out quality equivalence has not been established. The previous V5 archive remains available as a fallback.

## 1. Get the code

Use a CPU/login environment appropriate for small setup commands; perform downloads, extraction, compilation and large data checks on your site's CPU/data-transfer facilities, not a GPU allocation.

```bash
git clone https://github.com/TerryShenty/sam-audio-workspace.git
cd sam-audio-workspace
unzip qwen30b-4gpu-v6-portable-source.zip
cd qwen30b-4gpu-v6
python3 verify_package.py
```

Use this directory's instructions. The repository-root SOLOs quickstart is a different, SAM-only workflow.

## 2. Prepare a Linux Python environment

Read `ENVIRONMENT.md`. Use Python3.12, a compatible CUDA PyTorch/torchvision/torchaudio/TorchCodec/Triton stack, a C++ compiler, and FFmpeg. Your cluster's supported stack must be checked; the recorded original environment is not a fresh-install lockfile.

```bash
python3.12 -m venv /YOUR_STORAGE/envs/qwen-sam
source /YOUR_STORAGE/envs/qwen-sam/bin/activate
python -m pip install --upgrade pip
# Install your site's mutually compatible CUDA PyTorch stack first.
python -m pip install -r requirements-runtime.txt
```

Install the official external SAM dependencies listed in `ENVIRONMENT.md` (dacvae, ImageBind, perception_models/core, and CLAP). Keep the bundled modified `sam_audio` source. Confirm imports and waveform decoding in your CPU environment before requesting GPUs. Do not copy another machine's compiled `.so` files.

## 3. Download the original model weights

Request access to the official SAM model using your own HF account, then sign in. Never commit or send your access token.

```bash
hf auth login
python download_models.py --model-root /YOUR_STORAGE/models
```

Qwen's requested revision is resolved to an immutable commit and checked against the known model-file SHA256 values. SAM uses a fixed official revision. If Qwen main has changed, supply the matching `--qwen-revision COMMIT`; a hash mismatch must not be bypassed. The package never initializes production from a research-trained checkpoint.

## 4. Download the dataset after upload completion

Dataset: https://huggingface.co/datasets/TerryShenty/qwen30b-sam-joint-111k

The release is incomplete until `UPLOAD_COMPLETE.json` is present. It contains111,068 training rows and817 validation rows, approximately350.57GB of payload. Allow additional space for model weights, checkpoints and one temporary data shard.

```bash
python download_data.py --destination /YOUR_STORAGE/joint_data
```

The tool pins the HF commit, checks each shard's SHA256 and size, extracts it safely, and removes only its downloaded temporary tar after extraction. Resulting directories:

```text
joint_data/
  data/                 # prompts, splits, identities, Qwen processor/AV caches
  sam_namespace/        # SAM mixture/video/clean caches
  validation_audio/     # original held-out mixture and GT audio
```

Original videos and obsolete SAM text caches are not required by this cached path. Do not globally replace historical path strings in identity metadata: they participate in cache keys. `VALIDATION_AUDIO_MAP.json` handles actual scoring-audio relocation separately.

## 5. Bind local paths and prepare CPU assets

Copy `paths.example.json` to `my_paths.json` and set all four paths to the actual directories on your cluster:

```json
{
  "data_root": "/YOUR_STORAGE/joint_data/data",
  "sam_namespace": "/YOUR_STORAGE/joint_data/sam_namespace",
  "qwen_model": "/ACTUAL_QWEN_DIRECTORY_FROM_DOWNLOAD_MODELS",
  "sam_model": "/ACTUAL_SAM_DIRECTORY_FROM_DOWNLOAD_MODELS"
}
```

```bash
python prepare.py --paths my_paths.json
```

This checks model hashes and data metadata, binds local scoring-source/audio paths, builds four CPU ATen extensions, checks the V6 contract, and creates local admission receipts. Old machine-specific inode/mtime receipts are not portable. CPU_READY does not mean GPU validation passed.

## 6. Run a bounded four-GPU qualification job

Adjust account, partition, GPU type, CPUs, memory and time to your site's policy. The template requests one node, four GPUs,16 CPUs,128GiB host memory, one hour, no requeue. This is a template, not proof that every destination workload fits.

```bash
export QWEN_PACKAGE="$PWD"
export QWEN_PYTHON=/YOUR_STORAGE/envs/qwen-sam/bin/python
export QWEN_OUTPUT=/YOUR_STORAGE/runs/qwen-check-001
export QWEN_MODE=qualify
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION train.sbatch
```

One `torchrun` starts four persistent ranks. Each rank processes two examples; global batch is8. There are no per-example jobs, job arrays, automatic renewals or speed-search sweeps. Slurm must provide `SLURM_JOB_END_TIME`; the memory watchdog expects cgroup v2 with a finite allocation limit.

Monitor every rank's logs, progress, physical GPU memory and host memory. The check includes demanding inputs, real parameter updates, a complete checkpoint save/restore and effective continuation. Only a successful check creates `PORTABLE_GATE.json`. Diagnose failures instead of disabling guards or repeatedly resubmitting unchanged code.

## 7. Start training after qualification succeeds

```bash
export QWEN_MODE=train
export QWEN_OUTPUT=/YOUR_STORAGE/runs/qwen-train-001
export QWEN_GATE=/YOUR_STORAGE/runs/qwen-check-001/PORTABLE_GATE.json
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --time=24:00:00 train.sbatch
```

The training loop has no epoch cap and reserves time before the original scheduler deadline for a complete checkpoint. It saves periodically (default900 seconds). The600-second checkpoint reserve and120-second step guard are conservative template values, not measured limits on your hardware. The first three cold-start warmup updates are real training updates and count toward the data position.

## 8. Resume a formal training checkpoint

Use a new output directory and the same qualified deployment:

```bash
export QWEN_OUTPUT=/YOUR_STORAGE/runs/qwen-train-002
export QWEN_RESUME=/YOUR_STORAGE/runs/qwen-train-001/checkpoints/step-REPLACE/COMPLETE.json
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --time=24:00:00 train.sbatch
```

Use the actual checkpoint path produced by the run. State includes model, optimizer, scheduler, EMA, all-rank RNG, sampler/data position, configuration and identity. Qualification/research checkpoints are not formal production initialization. Do not restore V5 runtime defaults into V6 or bypass identity checks.

## 9. Complete validation integration on the target machine

Read `METHOD_AND_VALIDATION.md`. `validation_run.run(...)` is the existing four-rank validation API. It requires a loaded codec and a genuine memory-approval receipt from longest padded inputs in the retained training state, including decoding/scoring. Batch1/2/4/8 is supported only after that qualification.

**There is not yet a one-command automatic817-validation CLI.** The collaborator must finish codec/memory integration and invoke this API. Until then, training output deliberately records `full817_validation_completed=false`. Do not report successful validation from CPU checks or source delivery.

## Source changes and debugging

`SOURCE_MANIFEST.json` binds this export. `EXPORT_CHECKS.json` records completed checks. Historical version-named modules are retained only where the selected runtime depends on them; they are not instructions to rerun old experiments. Changes require a fresh manifest and affected checks. Preserve logs and complete checkpoints, and never hot-edit code loaded by a running training process.
