# SAM Audio engineering workflow

**Engineering preview — the new unified entry is still under real end-to-end acceptance.**
Training and validation data are public. The obsolete validation `revision: null`
block is fixed in this release. See [DELIVERY_STATUS.json](DELIVERY_STATUS.json)
for accepted evidence and remaining work; publication does not imply production readiness.

On a Linux x86_64 server with Python 3.12 and two NVIDIA GPUs:

```bash
git clone https://github.com/TerryShenty/sam-audio-workspace.git
cd sam-audio-workspace
bash start.sh --directory "$PWD/sam-run" --wandb-entity YOUR_WANDB_ENTITY
```

Use your own Hugging Face account with approved access to
[facebook/sam-audio-large-tv](https://huggingface.co/facebook/sam-audio-large-tv)
and your own W&B account. Authenticate with the normal `hf auth login` and
`wandb login` tools if installed, or supply `HF_TOKEN` and `WANDB_API_KEY`
securely in your environment. Never put credentials in this repository or a command shared with others.
The data repositories do not require the owners' credentials.

The default command installs the included environment if needed, downloads a fixed
**3-video / 8-target / 13-prompt training selection** and **all 285 validation examples**,
prepares source-placement GT spans, creates new training and validation GPU caches,
exports a checked bundle, runs nine engineering updates with full validation at
half/epoch/final boundaries, and finalizes best/last checkpoint archives.
It does not launch all 93,942 training videos. Environment installation, model
access, storage and a working W&B login are prerequisites for successful execution.

The wrapper verifies a content-addressed source archive and its complete source
manifest before executing it. Source lives under `sam-run/.sam-source/` and run
outputs under `sam-run/workspace/`. Reuse the same command and directory to check
completed stages and continue. Unknown compute or upload outcomes stop for
reconciliation rather than starting a duplicate. Checkpoints are not deleted.

To verify the source download only, without installing dependencies or allocating GPUs:

```bash
bash start.sh --directory "$PWD/source-check" --source-only
```

The tested resource reference is two A100 40GB GPUs and 64GB host RAM. Full285
validation takes about ten minutes per pass in the tested environment. The BF16,
global-batch-two engineering profile differs from SOLOs production INT8 settings.
Longer 30–50 second training examples and other GPU types have not been accepted.
GPU availability, driver compatibility, installation time and model downloads vary by server.
The full public training archives total about **1.313 TB compressed**, before
extraction, models and feature storage; the engineering selection is much smaller.

## Cluster execution

Keep network transfers, hashing, preparation and export outside GPU allocations.
These commands use the same workspace and entry, and must run sequentially:

```bash
# Data transfer facility
bash start.sh --directory "$PWD/sam-run" --wandb-entity YOUR_WANDB_ENTITY --through download
# CPU allocation
bash start.sh --directory "$PWD/sam-run" --only prepare
# Two-GPU allocation
bash start.sh --directory "$PWD/sam-run" --only encode-training
bash start.sh --directory "$PWD/sam-run" --only encode-validation
# CPU allocation
bash start.sh --directory "$PWD/sam-run" --only export
# Two-GPU allocation
bash start.sh --directory "$PWD/sam-run" --only train
# Data transfer facility, after all training processes and allocation steps exit
bash start.sh --directory "$PWD/sam-run" --only finalize
```

The wrapper does not submit scheduler jobs. Request resources using your facility's
policy and actual representative timings. No CPU-only phase runs in a Slurm GPU
allocation. A plain GPU server can run the default command sequentially.

Explicit `SAM_PYTHON` and `SAM_FFMPEG_PREFIX` select an existing environment.
`--hf-hub-cache` selects a model cache. Verified raw-media/model reuse options are
available in the extracted workflow help; all reuse is recorded. Omit them for an
ordinary new-server download. GPU stages use the pinned offline model cache.

## Public data and scope

- [AVSpeech 93,942-video release](https://huggingface.co/datasets/TerryShenty/avspeech-93942/tree/c8bdd1fd47e269c55f0323f54a550d77022f1e92): 256,708 targets/residual pairs and 390,402 prompts.
- [MSDWild78 + EasyCom207 validation](https://huggingface.co/datasets/sam-audio-research/sam-audio-validation-285/tree/9b438aab4d7987166304693ad616ae3769798e3c): pinned media, GT audio and original prompts; anonymously verified.

GT training spans describe ten-second source placement, including internal
silence; they are not VAD labels. Validation GT audio is used for scoring, not
model conditioning. The engineering selection remains pinned to an earlier
immutable revision containing exactly those three selected composites.

Existing older archives remain as historical releases. This root entry selects
only the archive named in its current `bootstrap.py`; `training-source.tar.gz`
and older handoff instructions are not the current execution path.
