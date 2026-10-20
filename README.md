# SAM Audio workspace

Code and environment preparation for a portable research workflow. **The complete training handoff has not passed acceptance yet.** This repository contains no datasets, model weights or credentials.

## One-command preparation

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/start.sh -o /tmp/sam-workspace-start.sh && bash /tmp/sam-workspace-start.sh /data/sam-workspace
```

Replace `/data/sam-workspace` with a writable directory on a sufficiently large disk. The command verifies the source, prepares the fixed real validation media, installs an isolated environment, checks imports and inspects Hugging Face release metadata. It does not yet download the complete dataset or start training. Linux x86_64, Python 3.12, git, FFmpeg with shared libraries, curl and outbound HTTPS are required. GPU execution also requires a compatible NVIDIA driver.

See [HANDOFF.md](HANDOFF.md) for operator instructions and [DELIVERY_STATUS.json](DELIVERY_STATUS.json) for acceptance boundaries.

## Evidence and remaining work

- 55 source regression tests and four bootstrap tests pass.
- The Linux dependency repair passed real CPU video decoding, dependency checks and repeated invocation. It corrects metadata in the known decord wheel without changing its native binary. This was tested in an existing isolated environment; a complete fresh-machine installation remains unverified.
- A small single-A100 engineering test completed training, a new-process checkpoint restore and a further update, followed by finite inference. It used an older harness and a small SOLOs subset.
- Engineering checkpoints were uploaded to W&B and downloaded into fresh caches, with file sizes and SHA256 verified. Continuing training from those downloaded checkpoints is still pending.
- Three actual HF AVSpeech composites, eight targets and all 13 associated prompts passed download and CPU media/GT checks. Their full-duration two-GPU encoding and CPU cache export passed. The current public entry also completed two-rank checkpoint save/resume and half-epoch/epoch validation on one real MSDWild and one real EasyCom row, with online W&B logging.
- All 78 MSDWild and 207 EasyCom validation rows have a prepared media package. It is **not in the public repository**. Full validation feature preparation and GPU scoring remain unaccepted.
- The two-A100 entry-point train/resume and half-epoch validation passed on the engineering AVSpeech pilot. Automatic best/last retention, downloaded-checkpoint continuation and the complete one-command workflow remain unaccepted.

Portable BF16/global batch 32 differs from production SOLOs INT8/global batch 64. The xformers native-extension ABI warning remains unresolved; the bounded successful SAM path does not certify that extension.

## Data and results

[Selected AVSpeech data](https://huggingface.co/datasets/TerryShenty/avspeech-93942) is a separate HF repository. Check an immutable release manifest before using it; upload completeness must not be assumed. Raw media is not a ready training cache.

The default W&B destination is `terryshen066-fudan-university/sam-audio`. The operator must authenticate with an account authorized to write there. Scores include dataset counts, aligned SI-SDR, SI-SDR and SI-SDRi, together with checkpoint and training-step coordinates. Checkpoint archives retain summaries and per-sample scores. Actual engineering half-epoch and epoch scores have been logged online. Full-cohort scoring and downloaded-checkpoint continuation remain pending.

Resume requires the pinned base model plus saved training state. No automatic server cleanup is enabled. Original SAM Audio licensing is retained; dataset and pretrained-model access and licenses are separate.

## Fixed validation workflow

The one-command entry now prepares MSDWild78 + EasyCom207 first, independently of AVSpeech upload completeness. It downloads a pinned media archive, verifies every file and the original cohort identities, prompts, noise seeds and durations. The configured destination is the public organization repository `sam-audio-research/sam-audio-validation-285`; publication is still awaiting explicit approval, so no immutable revision is available yet. Until a verified revision is published and pinned, preparation stops with an explicit message rather than silently using different or incomplete validation data. Environment-only preparation remains available through `bootstrap.py --install` without `--prepare-validation`.

Formal `train_delivery.py` runs check the complete 285-row cohort before GPU initialization. They validate every half epoch and retain separate MSDWild and EasyCom summaries. Tiny maintainer tests must explicitly set `engineering_validation_subset: true` and are recorded as `ENGINEERING_SUBSET`; they cannot establish complete-cohort acceptance. The old AVSpeech100 and SOLOs/URMP validation are excluded. Downloaded raw media still require the pinned GPU feature-cache stage; that integration and real full-cohort scoring remain pending.

## Native media dependency check

Setup now decodes a generated PCM WAV and H264 video through torchaudio, TorchCodec AudioDecoder/VideoDecoder and decord. A successful import or working static `ffmpeg` command alone is insufficient. A real pilot exposed missing shared FFmpeg libraries; a CPU regression reproduced the failure and verified the existing shared runtime on 23 audio files and five videos before any replacement GPU test.

For an existing separate shared FFmpeg installation, `SAM_FFMPEG_PREFIX=/path/to/ffmpeg` is supported by `setup.sh` and `run.sh`; its `lib/` directory is configured before Python starts. This optional hook does not install FFmpeg or prove a fresh server is ready. The tested A100 node also required `NCCL_P2P_DISABLE=1`; automatic communication diagnosis remains to be integrated, and that setting is not claimed optimal for every server.
