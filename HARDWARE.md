# Choosing GPUs and servers

A **node** is one physical server. Cards in one node are locally visible CUDA
devices. Cards in different nodes communicate over the network; a GPU index on
one server cannot select a GPU in another server. Ask the administrator for the
number of servers, GPUs per server, GPU model and inter-server network.

Twelve H200 GPUs could be divided across servers or belong to one large server;
the count alone does not establish topology. Several dozen cards are commonly a
multi-server cluster, but the deployment still needs to be checked.

## What has actually run

| Setup | Real evidence | Limits |
|---|---|---|
| One server, two A100 40GB | New training and all 285 validation features; nine GT updates; three full validations; complete checkpoint CPU audit; cloud metric readback; automatic best/last archival | Engineering selection and 10/20-second examples; CPU/transfer/GPU phases separated |
| Two different physical servers, one L40S 48GB and one L40 48GB | Global world size 2; actual 10/20-second updates 0→2; checkpoint; new-process 2→3 update; full saved-state CPU audit | Mixed-hardware functional test over a slow link; no cross-node full285/W&B acceptance or scaling claim |
| One server, 1, 4 or 8 selected GPUs | CPU tests for planning, batching, coverage, receipts and strict resume guards | Actual GPU execution at these counts is not accepted |
| H200, 12 GPUs, more nodes, RDMA/InfiniBand | No direct hardware acceptance | Do not infer from A100/L40 results |

## One-server selection

The default is two GPUs. Use `CUDA_VISIBLE_DEVICES` to choose your assigned
physical cards outside a scheduler. For example, `CUDA_VISIBLE_DEVICES=2,5`
exposes those two cards as local CUDA devices 0 and 1. In Slurm, preserve its
assigned visibility and request exactly the GPUs the phase uses.

This release accepts 1–8 selected GPUs per server. A larger server can
expose a subset. A 12-card host can therefore start with two selected cards;
`--gpus 12` is not supported by this engineering entry.

| Selected training GPUs | Microbatch per GPU | Default global batch | Accumulation | Epoch capacity for nine updates | Training encoder GPUs |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 2 | 2 × 7 updates | 1 |
| 2 | 1 | 2 | 1 | 2 × 7 updates | 2 |
| 4 | 1 | 4 | 1 | 3 × 4 updates | 3 |
| 8 | 1 | 8 | 1 | 5 × 2 updates | 3 |

The tiny training selection has only three unique mixtures, so its encoder uses
at most three GPUs. Validation encoding, training and scoring use the selected
count. A four/eight-GPU Slurm workflow must allocate three GPUs for this training
encoder, then four/eight for validation encoding and training. Do not leave
allocated workers idle merely to preserve a uniform request across phases.

`--global-batch` must be a positive multiple of the selected count. Changing
batch changes updates per epoch, validation boundaries and training trajectory.
Four/eight cards do not retain the two-card step4/step7 validation schedule and
may trigger more full validations. Do not extrapolate speed or wall time from
the two-card reference.

After extracting the source, these inspections do not launch training:

```bash
bash start.sh --gpu-plan --gpus 4
bash start.sh --check-gpus --gpus 4
```

`--gpu-plan` needs no Torch or installation. `--check-gpus` needs the installed
environment; the shell entry may install it if absent. To avoid that installation,
use the already-installed Python directly with `tools/standalone_workflow.py`.

## Memory and compatibility

DDP keeps a model replica on each card: four 48GB GPUs do not provide a single
192GB model-memory pool. The single-server entry checks visible count, distinct
UUIDs, CUDA BF16 capability, homogeneous selected devices and at least 22 GiB
reported capacity per card. That threshold allows nominal 24GB cards; it is an
admission rule, not an OOM guarantee or a measurement on every card model.
Identifiable MIG partitions and nested torchrun invocation are rejected.

There is no automatic batch enlargement, FP32 fallback, FSDP/ZeRO or elastic
checkpoint resharding. Host memory, driver/library compatibility, occupied VRAM
and input duration still matter. Longer 30–50-second training examples have not
been accepted. More GPUs require a new measured host-memory and timing estimate.

Full resume keeps the original world size, batch/accumulation, data/base identity,
optimizer, scheduler, EMA and per-rank RNG. A two-card checkpoint cannot silently
become an exact four-card resume. A new run or explicitly designed conversion is
required; loading only model weights is not complete continuation.

## Multiple servers

Use the explicit experimental launcher in [MULTINODE.md](MULTINODE.md), not the
single-server `--gpus` option. The implemented fixed-topology contract allows
2–16 nodes and 1–8 GPUs per node, but **only two nodes with one GPU each have
passed actual GPU testing**. More cards or nodes are implementation bounds, not
tested configurations.

The two-cloud-server experiment used about a 12 MB/s connection; initial model
broadcast dominated roughly ten-minute short phases. That result demonstrates
correct updates and resume, not efficient training. A fast intra-cluster network
and consistent devices should be evaluated on the actual target system. Existing
operator NCCL settings are preserved; the tested local fallback is not a universal
recommendation to disable P2P or InfiniBand.

## A100 versus L40S

L40S has 48GB GDDR6 ECC and no NVLink, according to the
[NVIDIA specification](https://www.nvidia.com/en-us/data-center/l40s/).
The tested A100 cards have 40GB per device. Capacity alone does not predict
training speed: local PCIe topology, peer access, memory bandwidth and the
selected workload matter. The current evidence does not compare two L40S cards
against two A100 cards on an otherwise identical single-server SAM run.

See [COMMUNICATION.md](COMMUNICATION.md) before choosing a multi-server rental.
