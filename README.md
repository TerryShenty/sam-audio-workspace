# SAM Audio workspace

Code and environment preparation for a portable research workflow. **The complete training handoff has not passed acceptance yet.** This repository contains no datasets, model weights or credentials.

## One-command preparation

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/start.sh -o /tmp/sam-workspace-start.sh && bash /tmp/sam-workspace-start.sh /data/sam-workspace
```

Replace `/data/sam-workspace` with a writable directory on a sufficiently large disk. The command verifies the source, prepares the fixed real validation media, installs an isolated environment, checks imports and inspects Hugging Face release metadata. It does not yet download the complete dataset or start training. Linux x86_64, Python 3.12, git, curl and outbound HTTPS are required. GPU execution also requires a compatible NVIDIA driver.

See [HANDOFF.md](HANDOFF.md) for operator instructions and [DELIVERY_STATUS.json](DELIVERY_STATUS.json) for acceptance boundaries.

## Evidence and remaining work

- 105 source regression tests and eleven bootstrap tests pass.
- The Linux dependency repair passed real CPU video decoding, dependency checks and repeated invocation. It corrects metadata in the known decord wheel without changing its native binary. A subsequent Linux test installed both a new Python virtual environment and a new shared FFmpeg prefix, repeated setup, and passed dependency/import/native-decoding checks. It reused package download bytes only, not an installed environment. GPU execution from that fresh environment remains unverified.
- A small single-A100 engineering test completed training, a new-process checkpoint restore and a further update, followed by finite inference. It used an older harness and a small SOLOs subset.
- Engineering checkpoints were uploaded to W&B and downloaded into fresh caches, with file sizes and SHA256 verified. Continuing training from those downloaded checkpoints is still pending.
- Three actual HF AVSpeech composites, eight targets and all 13 associated prompts passed download and CPU media/GT checks. Their full-duration two-GPU encoding and CPU cache export passed. The current public entry also completed two-rank checkpoint save/resume and half-epoch/epoch validation on one real MSDWild and one real EasyCom row, with online W&B logging.
- All 78 MSDWild and 207 EasyCom validation rows have a prepared media package. It is **not in the public repository**. Full validation feature preparation and GPU scoring remain unaccepted.
- The two-A100 entry-point train/resume and half-epoch validation passed on the engineering AVSpeech pilot. Automatic best/last retention, downloaded-checkpoint continuation and the complete one-command workflow remain unaccepted.

Portable BF16/global batch 32 differs from production SOLOs INT8/global batch 64. The xformers native-extension ABI warning remains unresolved; the bounded successful SAM path does not certify that extension.

## Data and results

[Selected AVSpeech data](https://huggingface.co/datasets/TerryShenty/avspeech-93942) is a separate HF repository. Check an immutable release manifest before using it; upload completeness must not be assumed. Raw media is not a ready training cache.

The default W&B destination is `terryshen066-fudan-university/sam-audio`. The operator must authenticate with an account authorized to write there. Scores include dataset counts, aligned SI-SDR, SI-SDR and SI-SDRi, together with checkpoint and training-step coordinates. Checkpoint archives retain summaries and per-sample scores. Actual engineering half-epoch and epoch scores have been logged online and independently matched against local results. Full-cohort scoring and downloaded-checkpoint continuation remain pending.

Resume requires the pinned base model plus saved training state. No automatic server cleanup is enabled. Original SAM Audio licensing is retained; dataset and pretrained-model access and licenses are separate.

## Fixed validation workflow

The one-command entry now prepares MSDWild78 + EasyCom207 first, independently of AVSpeech upload completeness. It downloads a pinned media archive, verifies every file and the original cohort identities, prompts, noise seeds and durations. The configured destination is the public organization repository `sam-audio-research/sam-audio-validation-285`; publication is still awaiting explicit approval, so no immutable revision is available yet. Until a verified revision is published and pinned, preparation stops with an explicit message rather than silently using different or incomplete validation data. Environment-only preparation remains available through `bootstrap.py --install` without `--prepare-validation`.

Formal `train_delivery.py` runs check the complete 285-row cohort before GPU initialization. They validate every half epoch and retain separate MSDWild and EasyCom summaries. Tiny maintainer tests must explicitly set `engineering_validation_subset: true` and are recorded as `ENGINEERING_SUBSET`; they cannot establish complete-cohort acceptance. The old AVSpeech100 and SOLOs/URMP validation are excluded. Downloaded raw media still require the pinned GPU feature-cache stage; that integration and real full-cohort scoring remain pending.

## Native media dependency check

Setup now decodes a generated PCM WAV and H264 video through torchaudio, TorchCodec AudioDecoder/VideoDecoder and decord. A successful import or working static `ffmpeg` command alone is insufficient. A real pilot exposed missing shared FFmpeg libraries; a CPU regression reproduced the failure and verified the existing shared runtime on 23 audio files and five videos before any replacement GPU test.

Setup automatically installs a pinned shared FFmpeg 7.1.1 runtime under `training/.runtime`, without sudo or changes to system packages. The installer verifies exact package sizes and SHA256 values, prevents concurrent installations and reuses matching completed installs. It also supports separate download and offline CPU-install phases on managed clusters. A fresh runtime install, repeat invocation and actual decoding of 23 audio files and five videos passed on Linux. A separate clean installation test then created both the Python environment and native prefix from scratch, passed repeated setup and actual fixture decoding in 4 minutes 48 seconds. Package download caches were reused; an empty network-cache installation duration and fresh-environment CUDA execution were not tested. An existing shared runtime may instead be selected with `SAM_FFMPEG_PREFIX=/path/to/ffmpeg`.

The multi-GPU launcher now runs a bounded allreduce/DDP/two-update probe before training. If the default transport fails, it first terminates the entire probe process group, then may try disabling NCCL P2P. Explicit operator settings are preserved. Missing rank receipts or both failures stop startup; training itself is never automatically retried. Regression tests passed, but this new automatic launcher still awaits its own GPU test. The earlier A100 diagnosis established the workaround only on that tested environment, not as a universally optimal setting.


## Best and last checkpoint handoff

`tools/archive_best_last.py --run-dir /path/to/completed-run` inspects the terminal run and plans both roles. `--upload` executes the verified archival after training has ended. Formal selection requires complete MSDWild78 and EasyCom207 summaries and uses their equal-weight mean aligned SI-SDR. The last role comes from the recorded latest successful checkpoint, not filename ordering. Engineering subsets require an explicit flag matching their saved configuration.

The helper keeps separate best/last receipts, verifies unchanged local content before reuse, and refuses to retry an unresolved upload. It never deletes server files. On Slurm, transfers must be run after the GPU allocation ends. A real engineering-pilot audit reconciled both existing cloud receipts and repeated without any new upload. The new helper's fresh live-upload path and automatic end-of-training integration still need acceptance; content verification alone is not resumed training.


After an independently prepared dataset/cache and environment are ready, run.sh now invokes a completion wrapper. On standalone servers it waits for all training processes to exit before verified best/last archival. Explicit offline logging or a Slurm GPU allocation produces a deferred archival record; those cases still require a subsequent transfer phase. This integration passed local subprocess regressions and real existing-receipt reconciliation, but has not yet passed its own GPU/fresh-upload end-to-end test. The one-command preparation entry above still does not launch a full experiment.


## Preparing selected AVSpeech audio

After `tools/download_selected_avspeech.py` has verified a pinned selection, `tools/prepare_selected_media.py --media /path/to/download --output /path/to/prepared` verifies its media against the download receipts, decodes the existing lossless target/residual audio, and checks that their sum matches the original mixture. It preserves every selected prompt and its category, subtype, target identity and provenance. Completed composites are checked before reuse; interrupted staging files are retained, and changed inputs or outputs cannot silently pass. Hashing and decoding must run on CPU before a Slurm GPU allocation.

The generic tool passed a real Linux CPU test on three HF composites, eight targets and 13 prompts: all 16 decoded target/residual waveforms exactly matched the earlier accepted preparation, and repeat invocation reused all three composites without decoding. Seven additional local regressions cover interruption, corruption and completeness. This remains an explicitly marked engineering subset. It does not prepare GPU features, GT-span anchors, full training data or launch training; the GPU stages and complete training integration remain pending.


## Combined download and CPU preparation

`tools/prepare_avspeech_workflow.py --directory /data/selection --revision COMMIT --ids ID1 ID2` combines immutable HF selection download with CPU audio preparation and verified source-placement GT support. Use `--all` instead of `--ids` only when every video, target and prompt in that frozen release is available. A workspace stays bound to its original selection. A failed download cannot trigger decoding, and failed or stale CPU/GT receipts cannot report readiness. Each successful CPU phase produces `support/train_rows.json` with the original prompts and verified GT intervals; GPU cache and GPU anchor acceptance remain pending. Repeating checks existing content before reuse; no server files are deleted.

Managed clusters can run `--phase download` on their transfer facility followed by `--phase prepare` on a CPU allocation. These real split phases passed on the three-video pilot, including a second CPU invocation that reused all completed composites. The transfer phase reused existing verified media and contacted HF for the pinned metadata; it was not an empty-cache bulk-download test. The combined standalone mode passed local control-flow regressions only.

The root bootstrap accepts `--install --prepare-avspeech --avspeech-revision COMMIT` plus either `--avspeech-ids ID1 ID2` or `--avspeech-all`; after environment checks it invokes this same combined tool. `start.sh` forwards these optional arguments and retains the fixed validation gate. This new bootstrap wiring passed mocked-stage/shell regressions; its full fresh-machine execution is not yet accepted. GPU feature preparation, GT-span conditioning and actual training remain separate pending stages. Nothing in this command certifies a completed training handoff.

## Training GT support preparation

`tools/prepare_training_spans.py --media /path/to/download --prepared /path/to/prepared --output /path/to/support` audits the original AVSpeech supervision rule on CPU: each source occupies one ten-second interval at its composite offset; simultaneous targets use the whole ten seconds. Silence within that interval is retained. This is source-placement supervision, not inferred voice activity. All prompts, categories, target identities and the rendered video/crop are preserved; validation inputs are not changed.

The tool checks frozen recipes, download receipts, prepared target/residual hashes and exact zeros outside temporal support. Original recipe identities and archived JSON hashes are recorded separately because public archival replaces host paths with provenance URIs. Changed media, redirected GT paths or modified prompts invalidate readiness.

A real Linux CPU test passed on three HF composites, eight targets and all 13 prompts, including a repeat invocation and an injected out-of-support waveform rejection. All 13 official anchor tensors exactly matched the historical AVSpeech conditioning implementation. The tool remains separately callable and is now run automatically after audio preparation. A new real CPU integration test completed both stages and a repeat invocation in seven seconds, matching all 13 prompt records and all eight targets against the accepted standalone result. Four coordinator regressions bring the source suite to 105 tests. This new test reused verified downloaded media; it did not repeat the download or exercise GPU forward or formal GT-conditioned training. It does not certify full training readiness.
