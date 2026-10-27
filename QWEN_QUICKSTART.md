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


## Output files and sharing results
Default formal output: `<root>/runs/train/` (override with `--output`).
- `training-rank0.csv` through `training-rank3.csv`: recorded local loss, gradient norm, step and approximate processed epochs. Warmup updates are included in step numbers but not CSV rows; resumed epochs are approximate.
- `VALIDATION817-summary.csv`: overall and per-dataset means, sample counts, training step, checkpoint identity and measured validation seconds (excluding the preliminary memory probe).
- `VALIDATION817-samples.csv`: all817 per-example scores and identities.
- `VALIDATION817.json`: authoritative complete-coverage validation receipt.
- `checkpoints/step-*/COMPLETE.json` and rank state files: complete resumable checkpoints. CSV files do not replace checkpoints.
Send the summary CSV and training-rank0.csv for a quick review; include all four training CSVs to investigate rank differences. Validation CSVs are generated only after complete817 coverage. Missing validation CSV does not mean a passing validation.
Validation now runs every half epoch based on committed global sampler progress (111,068 rows; global batch8), plus final validation when the final state has not already been scored. Boundaries occur after updates6,942,13,884,20,826,...; sample-based thresholds avoid cumulative rounding drift. Each boundary saves a complete checkpoint, runs all817 with the same scoring protocol, then resumes training. Fixed walltime takes priority near the deadline. Validation receipts are under `validation/step-*.json`, with matching `-summary.csv` and `-samples.csv`. The final summary remains `VALIDATION817-summary.csv`. A resumed checkpoint at an unfinished boundary is validated before further updates. No model, loss, optimizer, V6 kernel or validation solver change is made by this scheduling update. Target GPU execution remains pending.
