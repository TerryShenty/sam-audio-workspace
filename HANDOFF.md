# Server handoff

**Current release: preparation and testing only. Do not start a full experiment yet.**

## What the operator runs

Choose a writable absolute directory on a disk with enough free space:

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/start.sh -o /tmp/sam-workspace-start.sh && bash /tmp/sam-workspace-start.sh /data/sam-workspace
```

Replace only the final path. No GitHub login is required. This downloads and verifies the code, prepares the fixed real validation media independently of AVSpeech upload progress, installs an isolated Python environment, checks actual imports and records HF data status. It does not use sudo, change system Python, download the complete media, launch a long experiment or delete data. Retry the same command after an interrupted installation. Use a new directory when changing source versions; existing modified source is protected.

## One-time server prerequisites

Linux x86_64, Python 3.12, git, curl and outbound HTTPS. Training requires compatible NVIDIA hardware and driver. The entry reports missing prerequisites; Python packages cannot install the host driver or grant server permissions.

The operator uses their own W&B account with write access to `terryshen066-fudan-university/sam-audio`. The training logger and artifact uploader share this default destination. An existing login is sufficient only if it has that access. Never put API keys in GitHub or chat. W&B records results; it does not grant access to the GPU server.

## Data readiness

Training source: [TerryShenty/avspeech-93942](https://huggingface.co/datasets/TerryShenty/avspeech-93942). The intended release contains 93,942 composites, 256,708 targets and 390,402 prompt rows. Publication has been in progress; inspect a pinned manifest, not just the repository title.

A pilot of three actual HF composites, eight targets and 13 prompt rows passed download integrity and CPU media/GT checks. Its two-GPU full-duration encoding and CPU cache export passed. The current public entry trained two steps, saved full state, restored it in new processes and continued to step eight. Half-epoch (step four) and epoch (step seven) validation each scored one real MSDWild and one real EasyCom sample, with online W&B logging. This is an explicitly marked engineering subset, not full-cohort or model-quality acceptance.

Validation is real MSDWild 78 plus EasyCom 207. A complete media package has been audited and prepared separately, but is **not publicly downloadable through this handoff yet**. It preserves prompts, original per-sample noise seeds and 10/20-second durations. Media availability does not establish validation feature or scoring acceptance. The training dataset is not a substitute for this held-out cohort.

Pretrained base weights are obtained separately under their access terms. Resume checkpoints contain trainable parameters and optimizer, scheduler, RNG and data-position state; they also require the pinned base model and matching configuration.

## Results and recovery

The logger is configured to record each dataset separately, with sample count, aligned SI-SDR, SI-SDR and SI-SDRi, plus checkpoint and global training step. Per-sample results are retained with summaries in checkpoint archives. Independent cloud readback matched all metrics, counts and coordinates of both engineering validation records. Live full-cohort logging remains unverified.

`tools/upload_results.py` verifies the uploaded engineering artifact by downloading it and checking its content manifest. Independent receipts preserve both best and last references. `tools/restore_checkpoint.py` requires an immutable artifact version and a trusted upload receipt, verifies every file and refuses to overwrite an existing destination. Real engineering uploads and fresh-cache restores passed; GPU continuation from downloaded files is still pending.

`tools/select_best_checkpoint.py` supports an explicit equal-weight mean across datasets, requires complete expected cohort counts and retained matching checkpoints, and rejects missing or nonfinite scores. Automatic best-retention integration is implemented; real GPU and fresh-upload acceptance remain pending. Half-epoch save/validation triggering passed an actual two-rank engineering run; complete 285-row runtime acceptance is still pending.

No automatic server cleanup is enabled. Keep data and checkpoints until the necessary artifacts have been verified and successfully used for recovery.

## Maintainer acceptance

Passed: 101 source tests and eleven bootstrap tests; guarded decord metadata repair with actual before/after CPU decoding and dependency checks; older single-A100 engineering train/resume/inference; real engineering W&B content roundtrips; the actual HF pilot's CPU checks; full 285-row validation media preparation.

Remaining: GPU execution from the fresh environment; integrated AVSpeech feature-cache preparation; full real validation features/scoring; downloaded-checkpoint continuation; automatic best/last retention; and an integrated command that completes these stages. The current public training entry has passed the two-rank engineering pilot, including its explicit subset gate and half-epoch schedule.

The portable BF16/global-batch-32 path differs from the historical SOLOs INT8/global-batch-64 path. An xformers native ABI warning remains unresolved; no claim is made that its native extensions work.

## If preparation stops

Send `preparation-console.log`, `installation.log`, `MACHINE_REPORT.json` and `ENVIRONMENT_REPORT.json` when present, after reviewing private paths. Do not send credentials, `.netrc`, environment dumps or unrelated logs. The maintainer should repair the entry and supply a verified release; the colleague should not need to debug packages manually.

## Fixed validation workflow

The one-command entry now prepares MSDWild78 + EasyCom207 first, independently of AVSpeech upload completeness. It downloads a pinned media archive, verifies every file and the original cohort identities, prompts, noise seeds and durations. The configured destination is the public organization repository `sam-audio-research/sam-audio-validation-285`; publication is still awaiting explicit approval, so no immutable revision is available yet. Until a verified revision is published and pinned, preparation stops with an explicit message rather than silently using different or incomplete validation data. Environment-only preparation remains available through `bootstrap.py --install` without `--prepare-validation`.

Formal `train_delivery.py` runs check the complete 285-row cohort before GPU initialization. They validate every half epoch and retain separate MSDWild and EasyCom summaries. Tiny maintainer tests must explicitly set `engineering_validation_subset: true` and are recorded as `ENGINEERING_SUBSET`; they cannot establish complete-cohort acceptance. The old AVSpeech100 and SOLOs/URMP validation are excluded. Downloaded raw media still require the pinned GPU feature-cache stage; that integration and real full-cohort scoring remain pending.

## Native media dependency check

Setup now decodes a generated PCM WAV and H264 video through torchaudio, TorchCodec AudioDecoder/VideoDecoder and decord. A successful import or working static `ffmpeg` command alone is insufficient. A real pilot exposed missing shared FFmpeg libraries; a CPU regression reproduced the failure and verified the existing shared runtime on 23 audio files and five videos before any replacement GPU test.

Setup automatically installs a pinned shared FFmpeg 7.1.1 runtime under `training/.runtime`, without sudo or changes to system packages. The installer verifies exact package sizes and SHA256 values, prevents concurrent installations and reuses matching completed installs. It also supports separate download and offline CPU-install phases on managed clusters. A fresh runtime install, repeat invocation and actual decoding of 23 audio files and five videos passed on Linux. A separate clean installation test then created both the Python environment and native prefix from scratch, passed repeated setup and actual fixture decoding in 4 minutes 48 seconds. Package download caches were reused; an empty network-cache installation duration and fresh-environment CUDA execution were not tested. An existing shared runtime may instead be selected with `SAM_FFMPEG_PREFIX=/path/to/ffmpeg`.

The multi-GPU launcher now runs a bounded allreduce/DDP/two-update probe before training. If the default transport fails, it first terminates the entire probe process group, then may try disabling NCCL P2P. Explicit operator settings are preserved. Missing rank receipts or both failures stop startup; training itself is never automatically retried. Regression tests passed, but this new automatic launcher still awaits its own GPU test. The earlier A100 diagnosis established the workaround only on that tested environment, not as a universally optimal setting.


## Best and last checkpoint handoff

`tools/archive_best_last.py --run-dir /path/to/completed-run` inspects the terminal run and plans both roles. `--upload` executes the verified archival after training has ended. Formal selection requires complete MSDWild78 and EasyCom207 summaries and uses their equal-weight mean aligned SI-SDR. The last role comes from the recorded latest successful checkpoint, not filename ordering. Engineering subsets require an explicit flag matching their saved configuration.

The helper keeps separate best/last receipts, verifies unchanged local content before reuse, and refuses to retry an unresolved upload. It never deletes server files. On Slurm, transfers must be run after the GPU allocation ends. A real engineering-pilot audit reconciled both existing cloud receipts and repeated without any new upload. The new helper's fresh live-upload path and automatic end-of-training integration still need runtime acceptance; content verification alone is not resumed training.


## Automatic finalization (not yet GPU accepted)

run.sh now waits for the training process group to finish, checks a fresh successful terminal receipt, and calls tools/finalize_run.py. On a standalone server in online W&B mode this invokes verified best/last archival. A per-output lock rejects duplicate launches. Failed or interrupted training does not trigger archival; upload errors preserve the training result and a separate workflow report. Historical invocation reports remain available.

On Slurm GPU allocations, finalization records ARCHIVAL_DEFERRED before checkpoint hashing or transfer. Run tools/finalize_run.py --run-dir on the transfer facility after allocation termination. Explicit offline/disabled W&B mode also records a deferred result. No file cleanup is enabled. This orchestration passed seven local regressions, including real subprocess exit/failure and stale-receipt refusal; actual existing pilot receipts reconciled twice without new uploads. The new wrapper has not run GPU training and its fresh live-upload path remains unaccepted. Publishing this code does not establish GPU or fresh-upload acceptance.


## Preparing selected AVSpeech audio

After `tools/download_selected_avspeech.py` has verified a pinned selection, `tools/prepare_selected_media.py --media /path/to/download --output /path/to/prepared` verifies its media against the download receipts, decodes the existing lossless target/residual audio, and checks that their sum matches the original mixture. It preserves every selected prompt and its category, subtype, target identity and provenance. Completed composites are checked before reuse; interrupted staging files are retained, and changed inputs or outputs cannot silently pass. Hashing and decoding must run on CPU before a Slurm GPU allocation.

The generic tool passed a real Linux CPU test on three HF composites, eight targets and 13 prompts: all 16 decoded target/residual waveforms exactly matched the earlier accepted preparation, and repeat invocation reused all three composites without decoding. Seven additional local regressions cover interruption, corruption and completeness. This remains an explicitly marked engineering subset. It does not prepare GPU features, GT-span anchors, full training data or launch training; the GPU stages and complete training integration remain pending.


## Combined download and CPU preparation

`tools/prepare_avspeech_workflow.py --directory /data/selection --revision COMMIT --ids ID1 ID2` combines immutable HF selection download with CPU audio preparation. Use `--all` instead of `--ids` only when every video, target and prompt in that frozen release is available. A workspace stays bound to its original selection. A failed download cannot trigger decoding, and failed or stale CPU receipts cannot report readiness. Repeating checks existing content before reuse; no server files are deleted.

Managed clusters can run `--phase download` on their transfer facility followed by `--phase prepare` on a CPU allocation. These real split phases passed on the three-video pilot, including a second CPU invocation that reused all completed composites. The transfer phase reused existing verified media and contacted HF for the pinned metadata; it was not an empty-cache bulk-download test. The combined standalone mode passed local control-flow regressions only.

The root bootstrap accepts `--install --prepare-avspeech --avspeech-revision COMMIT` plus either `--avspeech-ids ID1 ID2` or `--avspeech-all`; after environment checks it invokes this same combined tool. `start.sh` forwards these optional arguments and retains the fixed validation gate. This new bootstrap wiring passed mocked-stage/shell regressions; its full fresh-machine execution is not yet accepted. GPU feature preparation, GT-span conditioning and actual training remain separate pending stages. Nothing in this command certifies a completed training handoff.

## Training GT support preparation

`tools/prepare_training_spans.py --media /path/to/download --prepared /path/to/prepared --output /path/to/support` audits the original AVSpeech supervision rule on CPU: each source occupies one ten-second interval at its composite offset; simultaneous targets use the whole ten seconds. Silence within that interval is retained. This is source-placement supervision, not inferred voice activity. All prompts, categories, target identities and the rendered video/crop are preserved; validation inputs are not changed.

The tool checks frozen recipes, download receipts, prepared target/residual hashes and exact zeros outside temporal support. Original recipe identities and archived JSON hashes are recorded separately because public archival replaces host paths with provenance URIs. Changed media, redirected GT paths or modified prompts invalidate readiness.

A real Linux CPU test passed on three HF composites, eight targets and all 13 prompts, including a repeat invocation and an injected out-of-support waveform rejection. All 13 official anchor tensors exactly matched the historical AVSpeech conditioning implementation. Eight new regressions bring the source suite to 101 tests. This tool is separately callable: automatic workflow integration, GPU forward with these anchors and formal GT-conditioned training remain untested. It does not certify full training readiness.
