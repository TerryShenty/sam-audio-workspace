# Server handoff

**Current release: preparation/testing only. Do not schedule a full experiment yet.**

## What the operator runs
Choose an absolute folder on a disk with sufficient space. No GitHub login or invitation is needed.

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/start.sh -o /tmp/sam-workspace-start.sh && bash /tmp/sam-workspace-start.sh /data/sam-workspace
```

Replace only `/data/sam-workspace` with your writable disk directory. The entry downloads and verifies the code, creates an isolated Python environment, installs dependencies, imports the actual modules, and records the HF release status. It does not modify an existing system Python, use sudo, submit jobs, delete data or start a long GPU run. Repeat the same command after an interrupted installation. Edited/different source is protected: choose a new workspace when upgrading to a different source bundle.

## One-time server prerequisites
Linux x86_64; Python 3.12; git; FFmpeg; curl; outbound HTTPS. GPU execution later requires an NVIDIA GPU and compatible driver. Drivers and site permissions cannot be bundled inside Python. If these are absent, the entry names the missing requirement before installation. No administrator password should be shared with the researcher.

## Data and results
- Public data: https://huggingface.co/datasets/TerryShenty/avspeech-93942 . It is still UPLOADING; do not infer completion from the dataset title. Full release has 93,942 videos, 256,708 targets and 390,402 prompts.
- This command only reads release metadata, not terabytes of media. A finalized media-to-cache workflow and real validation export are still required before enabling full training.
- W&B destination: `terryshen066-fudan-university/sam-audio`. The operator should log in with their own account invited to that team, or supply an authorized key through the server's secret manager. Never commit or send keys through GitHub/chat. Existing W&B login may be reused only if it can write to that team.
- Pretrained base weights are downloaded separately under their access/license terms. They are not contained in this code package.
- No automatic server cleanup. Keep the last complete resume checkpoint and best checkpoints until uploaded artifacts have been verified and successfully restored.

## If preparation stops
Send the researcher `preparation-console.log`, `installation.log`, `MACHINE_REPORT.json` and `ENVIRONMENT_REPORT.json` when present, after checking for private server paths. Do not send credentials, `.netrc`, environment dumps or unrelated logs. No need to manually rerun individual pip commands or debug Python packages.

## Maintainer acceptance still required
Clean Linux dependency installation/imports; exact source/fast-path selection; raw media/GT/crop-to-cache integration; MSDWild/EasyCom validation export; real GPU short train/validate/checkpoint/resume; W&B upload and restore. These are maintainer tasks, not work delegated to the colleague. The initial clean installation test reached its 20-minute limit while downloading PyTorch; it has NOT passed. A persistent CPU preparation job and dependent two-L40S short test are queued. The GPU test checks real updates, checkpoint restart and conditional audio generation on a small SOLOs TRAIN subset; it does not certify the complete AVSpeech data/validation workflow. Installation and multi-GPU results will be published after verification.

Completed data archives can be reused on retry after their file sizes and SHA256 are rechecked. This does not bypass unfinished HF media/GT or certify the training pipeline.

Checkpoint uploader now writes a size/SHA256 manifest and verifies SDK-downloaded content before a success receipt. Failed upload/corrupt downloads mark the upload run failed. Six local tests pass, including mocked SDK errors; live W&B and model resume remain untested. Download verification may use SDK cache and needs temporary disk space. Pending GPU tests remain pinned to earlier56ecd source.

Cached trainer initializes tuning on CPU and detaches unused frozen encoders before moving the training model to GPU; codec is retained separately for validation. Two ordering/error tests pass; actual CUDA memory/real entry-point acceptance remains pending. Existing pending smoke test is still pinned to56ecd and does not certify this release.

CPU export/preflight now validate all catalog/component/GT links against the file list, reject empty targets/prompts, duplicate identities, conflicting variants and noncanonical destinations. Seven contract/export fixture tests pass; these do not establish real tensor correctness or raw-to-cache readiness. Training entry does not yet automatically orchestrate the full CPU preflight.

Training terminal status repair: setup/training exceptions now write FAILED and finish W&B with exit_code=1. Tracking cleanup errors do not replace the original training error. Three mocked lifecycle tests pass; live GPU and W&B acceptance remain pending.

Validation compatibility: manifest must retain original noise_seed and latent_frames. Original packed forward-feature files supported with SHA256 verification. 20s inputs are no longer silently treated as10s. Real GPU evaluation remains untested; missing78 historical MSDWild feature files still block complete cohort export. Selected media recovery is verified separately, not model acceptance.

CPU export producer: tools/validation_export_plan.py translates historical285 inputs to relative bundle paths and explicit source/destination mapping. Exact78/207 cohort, original feature hashes, seeds and duration required. --plan-only is explicitly unverified. Export fragment must join training files and a source audit before export_bundle; it is not an automatic complete pipeline. Tiny full-export integration passes; real78missing features still block acceptance.
