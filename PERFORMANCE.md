# Measured native two-GPU speed — 2026-10-20

The public training runtime completed real SAM training on one physical server
with **two L40S 48GB GPUs**. Repeated continuous windows averaged
**0.305 seconds per optimizer update**, or **6.08 real targets/second**.
This is a small engineering workload, not a claim of globally optimal training.

## Conditions and results

All runs used the same public HF13 GT selection, full 10/20-second training inputs,
BF16, microbatch 1 per GPU, global batch 2 and seed 2026. Training used cached features,
SDPA, fused AdamW, no compilation and no gradient checkpointing. The environment
was freshly installed; verified SAM/T5 bytes and the public-data feature bundle
were explicitly copied. No old training checkpoint was used.

| Run | Updates | Warmup excluded | Timed updates | Seconds/update | Real targets/second |
|---|---:|---:|---:|---:|---:|
| Baseline repeat A |112|14|98|0.310|5.99|
| Baseline repeat B |112|14|98|0.301|6.17|

The two runs differ by 3.05% in throughput. Each contains 14 measured
seven-update windows: the trainer flushes at each epoch boundary even when the
requested log interval is 14. Across both runs, 196 timed updates processed 364 real
targets. Padding is excluded from target counts. Warmup, model loading, state
hashing, validation and checkpoint writes are excluded from these windows.
Whole processes took 127.0 and 121.7 seconds, including initialization and saving.
Peak Torch allocation was about 9.19 GiB per card. This does not prove memory safety
for 30–50-second inputs or different batch sizes.

Both ranks performed finite GT-conditioned forward/backward and nonzero Adam
updates. Every run saved complete model/optimizer/scheduler/EMA/data-position/RNG
state and passed CPU inspection of 484 model tensors, 484 Adam states and 484 FP32
EMA tensors. CPU state checks do not constitute an additional GPU resume test.
The per-window values and identities are in [PERFORMANCE.json](PERFORMANCE.json).

## Comparison with Longleaf

The earlier two-A100 public-workflow record contains four short training windows
at 0.373,0.377,0.419 and0.374 seconds/update with the same HF13/BF16/micro1/global2
recipe. The new L40S observations are in a comparable or lower latency range;
there is no observed abnormal slowdown on this native two-card host.

The older record has only four single-update windows, different window composition
and an earlier machine environment. It is not a repeated, simultaneous hardware
A/B. Do not interpret this comparison as a precise L40S-over-A100 percentage, the
fastest historical SOLOs configuration, or full-corpus production performance.

## Communication and the tested optimization

On this host, the default tiny NCCL probe timed out after 45 seconds. Its entire
process group was terminated, and the P2P-disabled fallback passed. All subsequent
SAM runs explicitly retained that tested host setting. This is not a recommendation
to disable P2P or RDMA on another server.

With three warmups and five measured repetitions, 128 MiB payloads reached median
4.10 GB/s broadcast and 3.59 GB/s all-reduce. These are synthetic payload rates,
not model throughput or a cross-node network measurement. Native DDP construction
with its normal initial synchronization took roughly 1.38–1.47 seconds.

An isolated candidate first checked actual values of every parameter and buffer
on both ranks, including frozen/nonpersistent state, before skipping initialization
synchronization. Its DDP construction fell to about 0.021 seconds. However, the
required state check itself took several seconds; the 28-step candidate measured
6.05 targets/second versus6.09 for baseline. After 28 steps, all 2,424 compared saved
tensors and scalar state matched exactly. **No steady-training speedup was
established, so the published default synchronization remains enabled.**

The earlier two-physical-server L40S/L40 test used a slow cross-region link and
proved updates/save/resume, not efficient scaling. Prefer a single server for
this small run. For larger multi-server runs, verify actual high-bandwidth
interconnect and measure matched steady-state scaling; see
[COMMUNICATION.md](COMMUNICATION.md) and [MULTINODE.md](MULTINODE.md).

## How to use the measured path

Use the existing two-card command in [README.md](README.md). This performance update adds no performance mode and enables no untested
optimization. A separate verified checkpoint-location fix is included; it was
not used by the speed experiments above. The normal entry still performs the nine-update check
and three complete 285-example validations. Those validation/archival stages were
not repeated in this speed experiment, so the table above is not end-to-end wall time.

H200/12-card systems, actual 1/4/8-card runs, fast-fabric multi-node scaling and
longest 30–50-second training inputs remain outside this acceptance.
