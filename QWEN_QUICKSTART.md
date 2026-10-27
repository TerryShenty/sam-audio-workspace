# Qwen30B + SAM Audio: one entry, one server, four 48GB GPUs

This entry selects the accepted V6 joint training implementation. Defaults: one node, four GPUs, two examples per rank/global batch8,16 CPUs,128GiB host RAM, no requeue. Actual GPU memory/driver compatibility and Slurm limits must match your server. The root SAM-only quickstart is a different workflow.

## 1. Install the private environment

Use Linux x86_64, Python3.12, git and a C++ compiler. On a cluster, use authorized CPU facilities with network access, not a GPU allocation. No sudo or system package changes are made.

```bash
curl -fL https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/qwen_quickstart.py -o qwen_quickstart.py
python3.12 qwen_quickstart.py setup --root /YOUR_STORAGE/qwen-sam
```

This downloads the hash-checked source, installs the original CUDA13 PyTorch stack and pinned upstream dependencies in a private environment, installs shared FFmpeg7, and checks imports plus actual waveform decoding. Your NVIDIA driver must support CUDA13. Missing compiler, unsupported ABI or unavailable package errors stop setup; no silent slower runtime is selected. Fresh installation and GPU execution on a recipient server are not yet accepted results.

## 2. Sign in once with your own HF account

Obtain access to `facebook/sam-audio-large-tv` through its official HF page. Public dataset access is separate and does not require model approval. Never share your token or another person's gated model access.

```bash
/YOUR_STORAGE/qwen-sam/environment/bin/hf auth login
```

## 3. Download and prepare everything

```bash
python3.12 qwen_quickstart.py prepare --root /YOUR_STORAGE/qwen-sam
```

This checks model access early, waits for the dataset release by refusing an incomplete release, downloads official Qwen/SAM weights and111,068train+817validation data, fills paths, builds CPU extensions, and extracts/tests the original frozen audio codec. Data payload is350.57GB; allow additional model/checkpoint/temporary space. Re-run after UPLOAD_COMPLETE.json is published if the release is still uploading. Existing download caches are reused.

## 4. Submit the four-GPU qualification

```bash
python3.12 qwen_quickstart.py qualify --root /YOUR_STORAGE/qwen-sam --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

This uses four persistent ranks in one one-hour Slurm job. It tests demanding training inputs, full-state resume, real retained-state validation memory, then all817 examples. The initial portable validation batch is1 per GPU; batch8 is not claimed safe in this changed joint state. PORTABLE_GATE.json requires actual complete validation. Diagnose a failed run before any repeat; the wrapper blocks duplicate submission paths.

## 5. Submit 24-hour training after qualification

```bash
python3.12 qwen_quickstart.py train --root /YOUR_STORAGE/qwen-sam --account YOUR_ACCOUNT --partition YOUR_PARTITION --gate /YOUR_STORAGE/qwen-sam/runs/qualify/PORTABLE_GATE.json
```

No epoch cap. It periodically saves complete state and reserves30 minutes near the original deadline for final817validation after saving the final complete checkpoint. VALIDATION817.json and its memory receipt contain coverage/scoring. Validation failure preserves the checkpoint, records failure and does not claim completion. There is no automatic renewal, blind OOM retry, dropped validation row or speed-search sweep.

Use a new `--output` for an intentional new run only. Check submission journals and the live scheduler before retrying an interrupted submission. Formal checkpoint resume remains available through the package's launch.py/train.sbatch interface with QWEN_RESUME; never use a qualification checkpoint as production initialization.

## Verification boundary

Automatic installer and validation wiring are implemented, not claims that a new server has passed. The existing-cluster CPU check exposed missing system FFmpeg libraries; an affected check uses the known working shared prefix. Actual target four-GPU and full817 GPU results remain pending. Model training weights, optimizer groups, data identities and V6 kernels are unchanged by this integration. The original production job is untouched.
