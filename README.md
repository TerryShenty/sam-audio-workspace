# SAM Audio workspace

See [HANDOFF.md](HANDOFF.md) for operator instructions and [DELIVERY_STATUS.json](DELIVERY_STATUS.json) for acceptance boundaries. Code only: no datasets, weights or credentials are included.

## One-command preparation

```bash
curl -fL https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/bootstrap.py -o /tmp/sam-audio-bootstrap.py && python3 /tmp/sam-audio-bootstrap.py --directory ./sam-audio-delivery
```

Add `--install` for isolated Linux Python 3.12 environment installation. The bootstrap verifies a pinned source archive and inspects data readiness. This is **preparation, not complete automatic production training**. The server owner must supply disk space, a compatible NVIDIA driver, model access and their own authorized W&B login.

## Verified and remaining

- 42 source regression tests; dependency coverage and actual CPU video-decoding probes included.
- A bounded Linux CPU test reinstalled the original decord 0.6.0 wheel, ran the automatic metadata-only correction, passed actual decoding and pip dependency checks, then passed a repeat invocation. Original metadata is backed up; the native binary is unchanged. This is a narrowly guarded workaround, not general ABI certification or a complete fresh-machine installation test.
- A separate single-A100 engineering test completed two training updates, saved and restored in a new process, continued one update and produced finite conditional inference output. It used a small SOLOs subset and older engineering harness, **not the latest full delivery entry**.
- Full AVSpeech raw-to-cache integration, latest entry-point GPU execution, multi-rank resume, real MSDWild/EasyCom scoring and live W&B artifact restoration remain unaccepted. Do not launch a full experiment from partial data.
- Validation metadata preserves the original per-sample noise seeds and 10/20-second durations. The intended real validation cohort is MSDWild 78 + EasyCom 207; planning/export code alone is not scoring acceptance.

Portable BF16/global batch 32 differs from production SOLOs INT8/global batch 64. The xformers native-extension compatibility warning is unresolved; a successful bounded SAM path does not certify that extension.

## Data and results

The selected-data release is https://huggingface.co/datasets/TerryShenty/avspeech-93942 . Inspect its current immutable manifest before use; upload completeness must not be assumed. Raw media is not a ready training cache.

W&B destination: entity `terryshen066-fudan-university`, project `sam-audio`. The runner logs in with their own authorized account. Keep secrets out of this repository. Checkpoint upload/download verification tooling preserves separate receipts for best and last artifacts; live end-to-end restoration remains to be tested. Resume checkpoints require the pinned base model in addition to saved trainable parameters and training state. No automatic server cleanup is enabled.

Source archive: `workspace-source-2014117cf5e7718c.tar.gz`. Verify with `sha256sum -c SHA256SUMS`; source contents have an additional SHA256 manifest. Historical notes inside the source bundle describe earlier tests; this README and HANDOFF provide the current acceptance status.

Original SAM Audio license is retained. Dataset and pretrained-model access and licenses remain separate.

Latest preparation changes: immutable W&B restore utility has passed real download/hash verification for engineering checkpoints (GPU continuation pending). Offline best-checkpoint selection requires explicit cohort counts and a dataset-equal criterion. Half-epoch save/validation triggering is now wired and unit tested; GPU execution of this change is pending. The queued short test remains pinned to the preceding source, so its results cannot certify these new changes.
