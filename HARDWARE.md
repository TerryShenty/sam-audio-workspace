# Choosing GPUs and servers

A node is one physical server. Cards within a node are local CUDA devices; cards in different nodes communicate over a network. Twelve or several dozen GPUs do not identify the number of servers, interconnect or usable GPU memory. Check the actual topology with the administrator.

## Measured configurations

| Setup | Accepted evidence | Boundary |
|---|---|---|
| One A100 40GB, fast profile | Same 384 full 10-second inputs as the accepted SOLOs reference; micro16/accumulation2/global32; repeated 50.2583 targets/s; full saved states. Separate public GT save8/new-process resume9 also passed. | Not H200 or long-input memory acceptance. |
| Two A100 40GB in one node, fast profile | Micro16/accumulation1/global32; two independent 480-update runs, 29,184 timed processing events; eight complete periodic checkpoints. Public micro2/global4 entry separately passed three updates with a full285 validation between updates2 and3. | Larger timing dataset has 384 distinct examples. Public13 is a functional check and cannot saturate the same batch. See FAST_PROFILE.md. |
| Two A100 40GB, original BF16 profile | New training and full285 features, nine updates, three full285 validations, full states, cloud metrics and automatic archival. | Historical complete engineering workflow; do not label the entire nine-update chain as newly rerun with the fast profile. |
| Two L40S 48GB in one node, original BF16 profile | Two 112-update repeats on public13, about 6.079 real targets/s at micro1/global2, full states. | Different recipe/input from the fast reference; not evidence that L40S is slower than A100. |
| Two physical nodes, one L40S 48GB and one L40 48GB, original BF16 profile | Real 0→2/save/new-process2→3 and complete CPU states. | Slow cross-region connection; fast profile, cross-node full285 and scaling remain unmeasured. |
| Four or eight selected GPUs | CPU planning, batch/coverage/receipt and resume rejection tests. | No current portable fast-profile GPU acceptance at these counts. |
| H200, twelve GPUs, more nodes, RDMA/InfiniBand | No direct acceptance. | Hardware availability or a successful two-card run does not establish throughput. |

## Select assigned GPUs

The single-server entry supports 1–8 selected GPUs and defaults to two. Outside a scheduler, `CUDA_VISIBLE_DEVICES=2,5` exposes those assigned physical cards as local devices0/1. Inside Slurm, preserve the allocation's visibility. A 12-card host can use a supported subset; `--gpus 12` is not supported by the engineering entry.

Inspect the selected recipe before compute:

```bash
bash start.sh --directory "$PWD/plan" --gpu-plan --gpus 2 \
  --performance-profile solos-fast-v1 --microbatch 2 --global-batch 4
```

The public bootstrap may fetch its small source package. The extracted planner itself needs no Torch, installation or GPU. `--check-gpus` requires the installed Torch environment.

The root cohort has only three unique training mixtures. Its training encoder uses `min(gpus,3)` workers; validation encoding, training and scoring use the selected count. Allocate the exact per-phase count. The planner records epochs, accumulation and batch size. Global batch must be divisible by `gpus × microbatch`; changing these changes the training trajectory and validation boundaries.

For saturated, explicitly prepared larger inputs, the measured recipe is micro16/global32: one card accumulates twice, two cards once. Four cards at micro16 require at least global64 and eight require at least global128; these are changed-batch experiments, not equivalent resumes or measured linear scaling. Keeping global32 requires smaller per-card batches. The 13-row public check limits microbatch to1/2/4.

## Memory and restoration

DDP replicates the model and does not pool memory. Micro16 reached approximately28.69GiB Torch allocation per A100 for full10-second inputs. This excludes some driver/NCCL overhead; it does not establish micro16 safety on nominal24GB GPUs or on20–50-second inputs. Use the smaller root recipe for an initial functional check and measure representative longer inputs before a larger run.

The entry checks visible count, distinct UUIDs, BF16 capability, homogeneous selected devices and at least22GiB reported memory; the capacity threshold is only an admission rule, not an OOM guarantee. Identifiable MIG partitions and nested torchrun are rejected. No automatic FSDP/ZeRO, elastic resharding or FP32 fallback is provided.

Complete resume keeps profile, world size, batch/accumulation, data/base identity, optimizer, scheduler, EMA and per-rank RNG. Changing a BF16 checkpoint to the INT8 fast recipe is not a complete compatible continuation. A different world or batch requires a deliberately defined new run or conversion, not silently reused state.

## Multiple servers

Increasing the single-server `--gpus` value does not reach another machine. Follow MULTINODE.md with independently verified inputs on each node and explicit ranks/network addresses. The fixed-topology launcher accepts2–16nodes and1–8GPUs per node, but only the original two-node/one-GPU-each BF16 save/resume test has actual cross-node acceptance.

Prefer measured local interconnects or a fast same-cluster fabric. Check actual NCCL transport; private IPs and same data-center labels are not bandwidth guarantees. Preserve functioning operator RDMA/P2P configuration. The Longleaf P2P workaround was specific to the tested host. See COMMUNICATION.md.

The migrated fast profile is separately documented in [FAST_PROFILE.md](FAST_PROFILE.md); previous BF16 scopes are historical and do not establish fast-profile multi-node or H200 throughput.
