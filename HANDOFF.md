# Server handoff

**Current release: preparation and testing only. Do not start a full experiment yet.**

## What the operator runs

Choose a writable absolute directory on a disk with enough free space:

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/start.sh -o /tmp/sam-workspace-start.sh && bash /tmp/sam-workspace-start.sh /data/sam-workspace
```

Replace only the final path. No GitHub login is required. This downloads and verifies the code, prepares the fixed real validation media independently of AVSpeech upload progress, installs an isolated Python environment, checks actual imports and records HF data status. It does not use sudo, change system Python, download the complete media, launch a long experiment or delete data. Retry the same command after an interrupted installation. Use a new directory when changing source versions; existing modified source is protected.

## One-time server prerequisites

Linux x86_64, Python 3.12, git, FFmpeg, curl and outbound HTTPS. Training requires compatible NVIDIA hardware and driver. The entry reports missing prerequisites; Python packages cannot install the host driver or grant server permissions.

The operator uses their own W&B account with write access to `terryshen066-fudan-university/sam-audio`. The training logger and artifact uploader share this default destination. An existing login is sufficient only if it has that access. Never put API keys in GitHub or chat. W&B records results; it does not grant access to the GPU server.

## Data readiness

Training source: [TerryShenty/avspeech-93942](https://huggingface.co/datasets/TerryShenty/avspeech-93942). The intended release contains 93,942 composites, 256,708 targets and 390,402 prompt rows. Publication has been in progress; inspect a pinned manifest, not just the repository title.

A pilot of three actual HF composites, eight targets and 13 prompt rows passed download integrity and CPU media/GT checks. The portable GPU feature-cache and training chain for this pilot has not run yet.

Validation is real MSDWild 78 plus EasyCom 207. A complete media package has been audited and prepared separately, but is **not publicly downloadable through this handoff yet**. It preserves prompts, original per-sample noise seeds and 10/20-second durations. Media availability does not establish validation feature or scoring acceptance. The training dataset is not a substitute for this held-out cohort.

Pretrained base weights are obtained separately under their access terms. Resume checkpoints contain trainable parameters and optimizer, scheduler, RNG and data-position state; they also require the pinned base model and matching configuration.

## Results and recovery

The logger is configured to record each dataset separately, with sample count, aligned SI-SDR, SI-SDR and SI-SDRi, plus checkpoint and global training step. Per-sample results are retained with summaries in checkpoint archives. These changes are unit tested; live full-cohort logging remains unverified.

`tools/upload_results.py` verifies the uploaded engineering artifact by downloading it and checking its content manifest. Independent receipts preserve both best and last references. `tools/restore_checkpoint.py` requires an immutable artifact version and a trusted upload receipt, verifies every file and refuses to overwrite an existing destination. Real engineering uploads and fresh-cache restores passed; GPU continuation from downloaded files is still pending.

`tools/select_best_checkpoint.py` supports an explicit equal-weight mean across datasets, requires complete expected cohort counts and retained matching checkpoints, and rejects missing or nonfinite scores. Automatic best-retention integration is not finished. Half-epoch save/validation triggering is implemented and unit tested, with runtime verification pending.

No automatic server cleanup is enabled. Keep data and checkpoints until the necessary artifacts have been verified and successfully used for recovery.

## Maintainer acceptance

Passed: 55 source tests and four bootstrap tests; guarded decord metadata repair with actual before/after CPU decoding and dependency checks; older single-A100 engineering train/resume/inference; real engineering W&B content roundtrips; the actual HF pilot's CPU checks; full 285-row validation media preparation.

Remaining: a complete clean installation; latest code on GPU; two-rank train/validation and complete resume; AVSpeech feature-cache integration; full real validation features/scoring; live metric logging; downloaded-checkpoint continuation; automatic best/last retention; and an integrated command that completes these stages. The pending two-A100 test uses a preceding immutable source and cannot certify later scheduling/logging changes.

The portable BF16/global-batch-32 path differs from the historical SOLOs INT8/global-batch-64 path. An xformers native ABI warning remains unresolved; no claim is made that its native extensions work.

## If preparation stops

Send `preparation-console.log`, `installation.log`, `MACHINE_REPORT.json` and `ENVIRONMENT_REPORT.json` when present, after reviewing private paths. Do not send credentials, `.netrc`, environment dumps or unrelated logs. The maintainer should repair the entry and supply a verified release; the colleague should not need to debug packages manually.

## Fixed validation workflow

The one-command entry now prepares MSDWild78 + EasyCom207 first, independently of AVSpeech upload completeness. It downloads a pinned media archive, verifies every file and the original cohort identities, prompts, noise seeds and durations. The configured destination is the public organization repository `sam-audio-research/sam-audio-validation-285`; publication is still awaiting explicit approval, so no immutable revision is available yet. Until a verified revision is published and pinned, preparation stops with an explicit message rather than silently using different or incomplete validation data. Environment-only preparation remains available through `bootstrap.py --install` without `--prepare-validation`.

Formal `train_delivery.py` runs check the complete 285-row cohort before GPU initialization. They validate every half epoch and retain separate MSDWild and EasyCom summaries. Tiny maintainer tests must explicitly set `engineering_validation_subset: true` and are recorded as `ENGINEERING_SUBSET`; they cannot establish complete-cohort acceptance. The old AVSpeech100 and SOLOs/URMP validation are excluded. Downloaded raw media still require the pinned GPU feature-cache stage; that integration and real full-cohort scoring remain pending.
