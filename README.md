# SAM Audio workspace

For server operators: see [HANDOFF.md](HANDOFF.md). A single preparation command and required credentials are listed there. This release is not yet accepted for full experiments.

**Preparation release — not yet GPU-certified.**

Code, environment inventory, data download/check tools and W&B integration are included in `workspace-source-2d49d79c553b5c40.tar.gz`. No training data, model weights or credentials are included.

```bash
git clone https://github.com/TerryShenty/sam-audio-workspace.git
cd sam-audio-workspace
sha256sum -c SHA256SUMS
tar -xzf workspace-source-2d49d79c553b5c40.tar.gz
cd training
```

# SAM Audio portable workspace — preparation release

**Not yet certified for end-to-end AVSpeech training.** This is an isolated delivery branch; running Longleaf production is untouched. It contains code, not data, checkpoints or credentials.

## What is included
- Training source, standalone distributed entry point (no Slurm required).
- Reference Linux Python 3.12.2 environment inventory from running SOLOs job 5571917, captured 2026-10-19. Torch reports 2.13.0+cu130.
- Resumable SHA256-verified data downloader, manifest checks, preflight and tests.
- W&B rank-zero metric logging and explicit checkpoint artifact uploader.
- Experimental selected-AVSpeech download helper, copied from the data release workflow.

## Important acceptance gaps
The existing environment can run production, but a fresh Linux installation has NOT been validated. Its pip check reports decord 0.6.0 unsupported. The freeze is evidence, not a certified installation lock. CUDA driver, decoder support and full train/validation/resume must be tested on the destination GPU.

The portable baseline uses BF16/global batch 32. Current SOLOs production uses INT8 frozen linear layers, attention/FFN LoRA and global batch 64. They are **not identical implementations**. Source parity and production fast-path portability remain pending; do not compare throughput as if identical.

AVSpeech selection currently has 93,942 videos, 256,708 targets and 390,402 prompts. Media upload is ongoing. Raw HF media is NOT the cached bundle consumed by train_delivery.py; cache construction and the real validation export are remaining integration work. Never start training from a partial upload.

The newly discovered public release is https://huggingface.co/datasets/TerryShenty/avspeech-93942 . At commit 2cb6d0877146f46049edad3b7d6aaa47f90576b8 it is UPLOADING: 390,402 prompts, 611 fully available videos, 2,183 GT groups. The public organization sam-audio-research has no listed AVSpeech dataset at this check. Public visibility and personal/organization ownership are separate concepts. Do not confuse this release with the original-source or cold-archive repositories.

## Intended deployment
1. Clone this public repository; no GitHub invitation is needed.
2. Install on Linux with Python 3.12 and an NVIDIA driver compatible with the chosen PyTorch wheel. `bash setup.sh` is an experimental installation attempt and stops on dependency errors.
3. Download the finalized HF release to a disk selected by its owner; verify hashes. Budget for media, extraction, caches and checkpoints rather than just archive size.
4. Produce and verify the cached bundle described below; run a short train/validate/save/resume check before a long run.
5. `wandb login` with the runner's OWN invited team account. Never put API keys in GitHub.
6. `GPUS=4 bash run.sh --data-root /data/bundle --model /models/large-tv --output runs/new-run` only after acceptance. GPU count and batch must match the measured hardware.

W&B defaults: entity `terryshen066-fudan-university`, project `sam-audio`, online mode. Without integration, scores remain local; this package integrates metric logging, but artifact upload is currently explicit:
`python tools/upload_results.py --run-dir runs/new-run --checkpoint final.pt`.
Retain the last complete resume checkpoint and a best checkpoint per declared validation criterion. The native checkpoint stores trainable weights plus optimizer/scheduler/EMA/RNG/data position; it also requires the pinned original base model (facebook/sam-audio-large-tv, revision b9fc94687ec044c570cbe30b8c28100cd056f1cb). It is not a standalone full base-model copy. Do not delete server files until artifacts have been verified and restored successfully. No automatic cleanup is enabled.

## Cached bundle contract
`bundle.json`: status VERIFIED_EXPORT, train_catalog, train_components, validation_manifest (relative paths).
Validation manifest: stores mapping name to path/signature; items with original stable index, dataset, store, refs, gt_audio, mixture_audio. This scaffold's validation adapter uses 10-second examples; AVSpeech temporal media may be 20 seconds. Training crop/GT alignment must be explicitly prepared, not assumed.
For the new AVSpeech run, use the declared real MSDWild/EasyCom validation splits. Do not reuse the old SOLOs/URMP cohort accidentally.

## Local checks
`python3 -m unittest discover -s tests -v`
`python3 -m compileall -q src tools`
These validate Python syntax and downloader behavior only, not GPU training.

## License
Original SAM Audio LICENSE is retained. Dataset and pretrained-weight licenses/access conditions remain separate. Dependencies retain their respective licenses.

## Single-command preparation entry

On the destination server:
```bash
curl -fL https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/bootstrap.py -o /tmp/sam-audio-bootstrap.py && python3 /tmp/sam-audio-bootstrap.py --directory ./sam-audio-delivery
```
This downloads a SHA256-verified pinned code archive, reports GPU/driver and required tools, and checks the live HF release at an immutable commit. It does NOT start training or download terabytes. Use `--install` with the same directory for an **experimental** isolated Python 3.12 install. Dependency failures are saved to installation.log. No sudo or cleanup. It cannot install the host NVIDIA driver, grant gated-model access or sign in to W&B on behalf of the server owner.

The source bundle is content-addressed and SHA256-verified. The root README contains the current HF release location.

2026-10-19 checks: anonymous public bootstrap download, archive SHA/extraction and live HF metadata lookup PASS. All 221 PyPI version pins exist; this does not prove dependency resolution, native binary compatibility or GPU operation.

## Preparation verification update
- Bootstrap is repeatable for byte-identical source and preserves an existing virtual environment; edited source triggers a clear stop without overwrite.
- setup installs the included package and imports the actual training/validation modules, rather than only checking version numbers.
- run.sh uses the local virtual environment automatically.
- SOURCE_MANIFEST.json enumerates included source/config/dependency files with SHA256.
- Longleaf decord metadata identifies a cp36 wheel; new installs resolve official Linux wheels. Existing production was not modified.
- Full clean Linux install, real GPU train/eval/resume, pretrained-model download, AVSpeech raw-to-cache preparation and validation bundle remain unverified/incomplete. No claim of complete one-command training.

Download recovery update: completed selected-release archives now have immutable-revision receipts. Reruns rehash outputs before skipping downloads. Incomplete archives still restart; full HTTP byte-range recovery is not claimed. Pending GPU acceptance remains on the earlier pinned source.
