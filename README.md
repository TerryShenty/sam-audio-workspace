# SAM Audio engineering workflow

This release provides a bounded **3-video / 8-target / 13-prompt training check**
with all **285 validation examples**. It does not start full 93,942-video training.
The tested single-server path uses two A100 40GB GPUs. A separate experimental
two-server launcher has also performed real training, checkpoint saving and
new-process resume on one L40S 48GB and one L40 48GB in different physical servers.

See [HARDWARE.md](HARDWARE.md) for GPU selection and the precise tested scope.
Having twelve or several dozen GPUs does not require using all of them.
Native two-L40S training speed and its repeated measured limits are documented
in [PERFORMANCE.md](PERFORMANCE.md).

## Start on one server

Use Linux x86_64, Python 3.12, an NVIDIA driver compatible with the installed
PyTorch CUDA runtime, and your own Hugging Face and W&B accounts. Your HF account
must have approved access to
[facebook/sam-audio-large-tv](https://huggingface.co/facebook/sam-audio-large-tv).
Training and validation data are public. Authenticate through the normal HF/W&B
login tools or securely supplied environment variables; never put tokens in Git.

```bash
git clone https://github.com/TerryShenty/sam-audio-workspace.git
cd sam-audio-workspace
CUDA_VISIBLE_DEVICES=0,1 bash start.sh \
  --directory "$PWD/sam-run" --gpus 2 --wandb-entity YOUR_WANDB_ENTITY
```

Outside a scheduler, `CUDA_VISIBLE_DEVICES` selects GPUs on this server. Replace
`0,1` with two GPUs assigned to you. Inside a scheduler allocation, keep its
assigned device visibility; do not select cards belonging to another allocation.

The command verifies the packaged source, installs its environment if needed,
downloads the pinned engineering training selection and full validation media,
prepares GT spans, builds new training and validation features, checks and exports
the bundle, performs nine updates, and archives the selected best/last checkpoints.
The default two-GPU profile scores all 285 examples at step 4, step 7 and step 9.
If best and last select the same checkpoint, one artifact serves both roles.

The same entry has completed this data-to-training-to-archive chain on Longleaf
with CPU, transfer and GPU stages separated. Environment/model/raw-download bytes
were explicitly reused; training and validation features and the run were new.
Cold anonymous source/media downloads and a fresh environment were tested
separately. A single uninterrupted invocation on an otherwise empty ordinary
server has not been observed. These scopes must not be conflated.

To verify only the public source, without installation, model downloads or GPU work:

```bash
bash start.sh --directory "$PWD/source-check" --source-only
```

To inspect a four-GPU plan before installation or GPU work:

```bash
bash start.sh --directory "$PWD/run4" --gpu-plan --gpus 4
```

This root command downloads/verifies the small source package if absent. The
extracted `start.sh --gpu-plan --gpus 4` itself performs no downloads or installation.
The four-GPU plan is CPU-checked; four-GPU execution has not been accepted.

## Continue and inspect

Source is stored under `sam-run/.sam-source/`; run outputs are under
`sam-run/workspace/`. `BOOTSTRAP_SOURCE.json` records the verified source path.
Use the same command and directory to check accepted stages and continue.
An unresolved compute/upload intent requires reconciliation before another
attempt. Do not delete intents to force a retry. Existing checkpoint files are
preserved. Changing source version, GPU count or global batch requires a new
workspace; full-state checkpoint continuation still enforces its saved identity.

The root entry does not allocate GPUs, rent servers, submit scheduler jobs or
automatically stop paid instances. Arrange those outside this program.

## Cluster stages

Use the same shared workspace and advance stages sequentially. Keep network,
hashing, CPU preparation and export outside GPU allocations.

```bash
# Transfer facility
bash start.sh --directory "$PWD/sam-run" --gpus 2 \
  --wandb-entity YOUR_WANDB_ENTITY --through download
# CPU allocation
bash start.sh --directory "$PWD/sam-run" --only prepare
# Two-GPU allocation, one server
bash start.sh --directory "$PWD/sam-run" --only encode-training
bash start.sh --directory "$PWD/sam-run" --only encode-validation
# CPU allocation
bash start.sh --directory "$PWD/sam-run" --only export
# Two-GPU allocation, one server
bash start.sh --directory "$PWD/sam-run" --only train
# Transfer facility after all training allocation steps have exited
bash start.sh --directory "$PWD/sam-run" --only finalize
```

Use the allocation size for each phase shown in [HARDWARE.md](HARDWARE.md).
The two-A100 reference used about 31 GiB host RAM, with a 64 GiB request.
Each full validation pass took about ten minutes. These are measured reference
values, not a promise for other hardware or networks.

Explicit `SAM_PYTHON` and `SAM_FFMPEG_PREFIX` select an installed environment;
`--hf-hub-cache` selects the model cache. Recorded raw/model reuse options are
available in the extracted workflow help. Omit reuse options for ordinary new
downloads. GPU stages use the pinned offline model cache.

## More than one server

The top-level `start.sh` is a **single-server workflow**. It does not combine
remote GPUs by increasing `--gpus`. The separate `tools/launch_multinode.py`
launcher requires prepared, independently verified inputs on every server,
explicit node ranks and reachable peer networking. See
[MULTINODE.md](MULTINODE.md) for the bounded two-server procedure and its limits.

## Data and limits

- [Full AVSpeech training release](https://huggingface.co/datasets/TerryShenty/avspeech-93942/tree/c8bdd1fd47e269c55f0323f54a550d77022f1e92): 93,942 videos, 256,708 target/residual pairs and 390,402 prompts; about 1.313 TB of compressed archives before extraction, models and feature storage.
- [MSDWild78 + EasyCom207 validation](https://huggingface.co/datasets/sam-audio-research/sam-audio-validation-285/tree/9b438aab4d7987166304693ad616ae3769798e3c): fixed original media, GT audio and prompts, anonymously verified by size and SHA256.

Engineering training stays pinned to the earlier immutable revision containing
its selected three composites. Source-placement GT spans include silence and
are not VAD labels. Validation GT is used for scoring, not conditioning.

This BF16 engineering profile differs from the SOLOs production INT8 profile.
The tests do not establish full-corpus quality, 30–50-second training memory,
H200 compatibility or twelve/dozens-of-GPU efficiency. The xformers native ABI
warning remains; the tested SAM path uses SDPA.

## Release boundary

The two-server save/resume and native two-L40S speed experiments used the accepted
152-file worker. This release adds its separately GPU- and CPU-verified terminal
checkpoint relocation fix. After an independently downloaded final checkpoint was
moved to a workspace with a different model directory, both A100 ranks restored
step 9 and performed a real step 10. The strict loader must verify identical SAM
weights/config and T5 identities; changing a path alone never authorizes a different
model. Full saved model, Adam, EMA, scheduler, data position and two-rank RNG state
passed CPU inspection, and the original checkpoint stayed unchanged.

This is an explicit constant-scheduler/warmup-50 engineering continuation, not
elastic world-size resume or a changed training recipe. Normal strict resume is
unchanged. See DELIVERY_STATUS.json for the exact scope and
[COMMUNICATION.md](COMMUNICATION.md) for the earlier slow-link limitation.
