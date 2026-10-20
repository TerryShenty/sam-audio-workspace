# Communication and speed

Single-server and multi-server training both use PyTorch DistributedDataParallel
(DDP). Each process owns one GPU and one model replica. On a single server,
communication uses the available local GPU interconnect; across servers it also
uses the network. More GPUs do not pool model memory or guarantee a faster run.

## What the two-server test established

The accepted pair was one L40S and one L40 in two different physical machines
and two different regions. NCCL logs selected `NET/Socket`, not RDMA. Observed
traffic was about 12 MB/s. The bounded SAM 0→2 phase took about 638 seconds;
fresh-process 2→3 resume took about 628 seconds. Both performed real finite
updates and saved auditable state. Those whole-process times include loading,
initial synchronization, training and saving; they are not steady-state training
throughput or a hardware comparison against the two-A100 reference.

The tiny all-reduce/DDP connectivity probe passed, but it did not expose the
large-model startup cost. Source inspection shows that the trainer uses the
default DDP initialization synchronization. PyTorch documents that this checks
shapes and broadcasts parameters/buffers. Combined with the slow measured link,
this is the main startup bottleneck hypothesis; the current receipts do not
contain a dedicated monotonic timer around that broadcast, so no exact fraction
or verified optimization speedup is claimed.

## Choose and measure the actual topology

For a small job that fits in one server, start there. For a multi-server job,
prefer the same data center with documented high-bandwidth interconnect and
verify the actual transport selected by NCCL. Private IP reachability alone does
not imply a fast training fabric. A faster GPU cannot fix a slow inter-server
link. H200 clusters must be checked for their real GPU-per-node layout, local
NVLink/NVSwitch and inter-node network rather than inferred from the card count.

Before a paid training allocation, prepare source, environment, model and input
bytes on CPU/transfer facilities. Then use a short allocation to inspect
`nvidia-smi topo -m`, test the selected interfaces and run the bounded launcher
probe on every rank. An administrator can also use NVIDIA's
[nccl-tests](https://github.com/NVIDIA/nccl-tests) on that exact allocation with
representative payload sizes. A small successful collective is a correctness
check, not a throughput acceptance.

For a speed comparison, record model loading, DDP construction, warmup, repeated
steady-state forward/backward/update, validation and saving separately. Use the
same samples, lengths, precision and global batch. Report the slowest rank,
per-card memory/compute, network transport and throughput; do not count startup
or a different batch profile as acceleration. Stop a test with missing ranks or
stalled progress rather than extending it blindly.

Keep operator-provided NCCL settings. Do not copy the cloud test's forced TCP
settings onto a working InfiniBand/RDMA cluster, or globally disable P2P because
one PCIe machine required a fallback. The trainer already uses `no_sync()` for
non-final gradient-accumulation microsteps. Disabling initial synchronization
without verifying every rank's actual parameter/buffer contents is not included
in this release; it can silently start inconsistent replicas.

## Current limits and next useful tests

No accepted H200 run, 4/8-card GPU run, fast-fabric scaling measurement or
multi-node full285 validation is available. The next useful paid test is a
bounded same-data-center comparison with initialization and steady-state
communication measured separately, not another cross-region repeat or an
unbounded large-cluster training run. The launcher does not rent or stop
instances: set a hard external lifetime/budget and verify provider shutdown.

References: [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
