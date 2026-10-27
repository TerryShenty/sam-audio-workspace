# Qwen30B + SAM Audio V6 automatic handoff

Target: one Linux x86_64 server with four approximately48GB NVIDIA GPUs,16 CPUs and128GiB host memory, subject to site policy and actual qualification. Training is per-rank2/global8 with the accepted V6 joint path. This is not multi-node training. Use QWEN_QUICKSTART.md in the repository for setup, own-account HF login, preparation and submission commands.

This revision adds an automatic private environment installer, early official-model access checks, CPU extraction/smoke decoding of the original SAM codec, and automatic complete817 validation. It preserves the model/optimizer joint training recipe and frozen AV cache identities. The original V6 speed result is oneA100/global2, not a four-card guarantee.

Qualification now tests longest training inputs and full checkpoint restoration, then screens validation memory with the retained training state and runs all817 held-out samples. It writes PORTABLE_GATE.json only when validation completes. Formal training saves its complete final checkpoint before validation and reserves30 minutes for validation inside the original walltime. Validation failure preserves the checkpoint and is reported as failure, never successful training completion. Full validation uses live weights, original seeds, midpoint16, original GT/mixture scoring and a coverage audit. Per-GPU batch1 is the conservative initial portable policy; batch8 has not been approved in this retained joint-training state. No automatic OOM retry or example dropping.

The new code path has source/CPU checks, not a target four-GPU acceptance result. Fresh environment installation and the actual817 GPU run must succeed on the recipient's server before claiming end-to-end completion. Existing queued production code is unchanged. Do not use qualification-trained state to initialize formal training.

Start here: [simplified commands](QWEN_QUICKSTART.md). Use `qwen30b-4gpu-v6-auto-source.zip`; the older portable ZIP remains a historical fallback.
