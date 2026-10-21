# The measured SOLOs fast training profile

Use `--performance-profile solos-fast-v1` to select the migrated, repeatedly measured SOLOs recipe. This is an explicit recipe choice: it changes frozen linear computations to INT8 row quantization while trainable parameters remain BF16. The original frozen BF16 weights are retained for validation. Full-state resume requires the same profile; do not resume a BF16-profile checkpoint as a fast-profile run.

The port includes the exact accepted row-quantization and INT8 GEMM kernels, contiguous Q/K/V before SDPA, native FP32 RMS normalization, per-transformer-block dynamic compilation and GEMM-only autotuning. Fused AdamW, gradient bucket views and gradient accumulation with `no_sync` were already present. Normal DDP initialization synchronization stays enabled; skipping it did not improve steady training in the earlier matched native test.

## What speed was reproduced

On one A100 40GB, the original accepted recipe measured 50.4336 and 50.3127 real training targets/second. The migrated code measured 50.4297 and 50.0880; pooled throughput was **50.2583 targets/second**, 0.228% below the original repeat mean. Both used the same 384 training examples, full 10-second inputs, microbatch 16, accumulation 2 and effective batch 32. The original long run used 405 updates/15 warmup; the port used two independent 120-update processes/24 warmup. This is evidence of reproducing the accepted recipe's speed, not a universal hardware maximum or bitwise trajectory comparison.

Each migrated repeat timed 3,072 real targets. Peak Torch allocation was about 28.69 GiB. Whole processes took 388.0 and 225.2 seconds, including model initialization, compilation and saving; the second explicitly reused the task's compile cache. These totals must not be substituted for the steady-state window measurement. Both complete checkpoints passed CPU state audits. Separate public GT training and new-process checkpoint continuation also passed with full 10/20-second inputs.

Later historical coordinate-descent, selective split-K and disabled epilogue-fusion trials did not establish a faster training configuration. The EXHAUSTIVE-search trial was interrupted when that research was paused; it is not an accepted faster version. Private diagnostic sample catalogs are not included in this release.

## Relation to the earlier four-L40S result

The historical four-L40S run measured143.0045 and143.7982 targets/s with micro16 and effective batch64. Its per-card batch used the same accepted fast combination, but GPU model/count and global batch differ from the new two-A100 global32 comparison. It is not a measured four-card result for this newly packaged portable entry. The new single-card measurement directly checks that the accepted fast computation was preserved; the new two-card measurement checks fixed-batch scaling on its actual topology.

A claim such as “sixteen epochs in24hours” also depends on the actual prompt count, duration distribution, validation frequency, compilation, checkpoint writes and queue time. Steady targets/s alone is not an end-to-end completion promise. Increasing global batch for more GPUs changes the optimizer trajectory; record that choice rather than presenting it as an exact continuation of a smaller-batch checkpoint.

## Public engineering check

The public root intentionally starts a bounded 13-prompt engineering check, not an unrequested full-dataset training run. Select the fast recipe explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash start.sh \
  --directory "$PWD/sam-fast-check" --gpus 2 \
  --performance-profile solos-fast-v1 --microbatch 2 --global-batch 4 \
  --wandb-entity YOUR_WANDB_ENTITY
```

This small check has mixed 10/20-second examples and only 13 rows. It cannot fill the same microbatch-16 workload that produced 50.26 targets/second. Its first compilation and full validation passes can dominate elapsed time. With microbatch 2/global 4 there are four optimizer updates per epoch; the nine-update check therefore spans three epochs and has different validation boundaries from the default microbatch-1 run. Use `--gpu-plan` first to inspect the selected plan. The default BF16 path remains available for a quick functional check; it is not the fastest steady training recipe.

## Train an explicitly prepared larger dataset

Use the same extracted source's `run.sh` with a verified complete bundle and a matching training configuration. `BUNDLE`, `CONFIG`, `MODEL`, `OUTPUT`, `EPOCHS` and `MAX_STEPS` below must name your prepared inputs and explicit training budget; the command does not silently use a private Longleaf dataset.

```bash
# One GPU: micro16 * accumulation2 = effective32.
GPUS=1 CUDA_VISIBLE_DEVICES=0 bash run.sh \
  --config "$CONFIG" --data-root "$BUNDLE" --model "$MODEL" \
  --output "$OUTPUT" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
  --performance-profile solos-fast-v1 --microbatch 16 --global-batch 32

# Two GPUs: micro16 * two ranks * accumulation1 = effective32.
GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash run.sh \
  --config "$CONFIG" --data-root "$BUNDLE" --model "$MODEL" \
  --output "$OUTPUT" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
  --performance-profile solos-fast-v1 --microbatch 16 --global-batch 32
```

Run this from the extracted source directory recorded in `BOOTSTRAP_SOURCE.json`, with its installed environment or explicit `SAM_PYTHON`, and your own W&B account. The low-level trainer enforces the verified bundle, model, GT/data/config and checkpoint identities. The fixed 13-row root deliberately limits microbatch to 1/2/4; do not pad a tiny engineering cohort into a claimed production-speed benchmark.

For four GPUs, microbatch16 requires an effective batch of at least64; eight requires at least128. That changes the optimizer trajectory and checkpoint data-position schedule. To keep effective32, the per-GPU microbatch must decrease, which can reduce per-card efficiency. Do not treat either as guaranteed linear scaling or silently change a checkpoint's saved world size/batch. DDP replicates the model and does not pool GPU memory. Micro16 was measured on full 10-second examples; 20–50-second memory at that batch is not established.

For multiple servers, use the separate verified-input multi-node procedure, not a larger single-host `--gpus` value. Choose a network with measured bandwidth/latency and inspect actual NCCL transport. Retain working RDMA/P2P settings rather than copying an operator workaround from a different host. The former two-server L40S/L40 test established functional save/resume on a slow link, not fast-profile scaling across nodes. H200/12-card/large-cluster performance needs its own measured acceptance.

## Extended stability gate

The release additionally requires two independent 480-update two-GPU runs at effective batch32. Each discards24 warmup updates and measures38 complete 384-example epochs:29,184 actual timed sample processing events total, drawn from384 distinct examples. Complete states are saved at120/240/360/480; checkpoint I/O and cold compilation are recorded separately. The two repeats and early/late windows must stay within5% before the draft is allowed to publish; negative two-GPU scaling blocks publication pending diagnosis.

## Current two-card result

On the same 384 distinct inputs and fixed effective batch of 32, two independent 480-update processes (24 warmup updates each, 29,184 timed sample processing events, 38 complete epoch windows per process) measured 71.9098 real targets/second (1.431 times the measured single card, 71.5% per-card scaling efficiency). This is measured SAM throughput; synthetic collective rates are reported separately.

The actual public bundle entry performed three fast-profile GT updates with a complete 285-example original-BF16 validation between updates 2 and 3; full checkpoint state, fixed noise and ten independently read cloud score fields passed. Reused model, environment and public feature bytes are explicitly recorded; this is not a new cold installation.
