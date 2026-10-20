# Experimental two-server training and resume

This is an administrator-facing, bounded training procedure for prepared inputs.
It is separate from the single-server raw-media-to-archive workflow. It does not
provision servers, distribute files, audit input bytes for you or stop cloud
billing. Those duties belong to the external operator/controller.

The accepted experiment used **two distinct physical servers, one L40S 48GB and
one L40 48GB**, one process/GPU per server. Both performed real SAM GT updates
0→2, saved a full checkpoint, then new processes restored it and performed 2→3.
The checkpoint model, Adam, EMA, scheduler, position, data/base identity and two
RNG records were independently audited. CPU RNG restoration was tested; CUDA
RNG storage does not establish replay equivalence. GPU restore checks sampled
parameters; the CPU audit checked the complete saved state.

This is not acceptance of H200, 12 GPUs, more than two nodes, cross-node full285
validation, cross-node W&B archival or an empty-server one-command deployment.
The training/scoring code has a fixed-world multi-node path, but its unexecuted
branches must not be described as tested.

## Prepare both servers

Use the same pinned source and environment, the same verified exported bundle,
the same SAM/T5 model bytes and an offline-capable tokenizer cache. The ordinary
workflow's `--through export` produces a checked bundle on a preparation system.
Stage copies through your approved transfer facility and independently check
every source/bundle/model file against its pinned size and SHA256 on each host.
Do not copy account tokens to another user or use a checksum string as a
substitute for actually verifying the files.

Keep the source, bundle, model, config, output and resume paths identical across
hosts. They may be separate local filesystems at the same absolute paths; the
current identity comparison includes path strings. A shared output filesystem
is not required. Copy the rank-zero checkpoint to the other host at the same
resume path and verify its size/SHA before starting new resume processes.

An external CPU preflight must produce a receipt with a common
`cpu_input_identity`: the SHA256 of canonical verified source/config/bundle/
SAM/T5 identities. Machine-local paths/stat details may be retained separately.
The receipts must agree across peers. The launcher compares that identity and
trainer metadata across ranks; it does not replace the CPU byte audit. The
published experimental launcher currently requires this operator-prepared
receipt rather than providing a general multi-host preparation controller.

Obtain distinct **physical machine IDs** from your cloud provider or cluster
administrator. A container ID, pod ID or hostname alone does not prove different
physical placement. Select a peer-reachable private address for node zero and
permit bidirectional peer traffic over the trusted training network. Test the
rendezvous and data-transport paths, not only SSH access. Do not expose an
unauthenticated training rendezvous to the public internet.

Configure the actual network interface for NCCL/Gloo if autodetection is wrong.
The accepted cloud test used TCP; its interface name and `NCCL_IB_DISABLE=1`
are not portable defaults. Do not disable a working RDMA path merely to imitate
that test.

## Launch one process per server

The following template starts the accepted **two-node / one-GPU-per-node** shape.
It requires prepared inputs and the installed environment; it is not an allocation
request. Run it on both hosts, with `NODE_RANK=0` on the first and `NODE_RANK=1` on
the second. Use assigned GPUs only. Set the variables to your own audited paths
and physical machine IDs before executing it.

```bash
# Set these on BOTH hosts; paths and the ordered ID list must match.
export SOURCE_DIR=/absolute/common/path/source
export SAM_WORK_ROOT=/absolute/common/path/run
export SAM_MODEL=/absolute/common/path/models/sam-audio
export HF_HUB_CACHE=/absolute/common/path/hf-cache
export MASTER_ADDRESS=PRIVATE_ADDRESS_OF_NODE_ZERO
export MACHINE_IDS_JSON='["PHYSICAL_MACHINE_ZERO","PHYSICAL_MACHINE_ONE"]'
# On each host separately, set NODE_RANK and THIS_MACHINE_ID to its assignment.

cd "$SOURCE_DIR"
export SAM_FFMPEG_PREFIX="$SOURCE_DIR/.runtime/ffmpeg"
source tools/media_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SAM_T5_REVISION=a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# This bounded engineering test deliberately has no cloud-metric acceptance.
export WANDB_MODE=disabled
export CPU_INPUT_IDENTITY=$(.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["cpu_input_identity"])' \
  "$SAM_WORK_ROOT/CPU_INPUT_READY.json")

.venv/bin/python tools/launch_multinode.py \
  --nodes 2 --gpus-per-node 1 --node-rank "$NODE_RANK" \
  --machine-id "$THIS_MACHINE_ID" --machine-ids "$MACHINE_IDS_JSON" \
  --master-address "$MASTER_ADDRESS" --master-port 29500 \
  --cpu-input-identity "$CPU_INPUT_IDENTITY" \
  --timeout 180 --output "$SAM_WORK_ROOT/probe" --phase probe
```

Check both probe processes exited successfully and inspect every rank receipt.
An agent exit code by itself is not evidence that SAM trained. After the probe
has finished on both hosts, start this bounded training phase on each host:

```bash
.venv/bin/python tools/launch_multinode.py \
  --nodes 2 --gpus-per-node 1 --node-rank "$NODE_RANK" \
  --machine-id "$THIS_MACHINE_ID" --machine-ids "$MACHINE_IDS_JSON" \
  --master-address "$MASTER_ADDRESS" --master-port 29501 \
  --cpu-input-identity "$CPU_INPUT_IDENTITY" \
  --timeout 900 --output "$SAM_WORK_ROOT/save2" --phase train -- \
  src/train_delivery.py --config configs/engineering-public.json \
  --data-root "$SAM_WORK_ROOT/bundle" --model "$SAM_MODEL" \
  --output "$SAM_WORK_ROOT/train" --epochs 2 --microbatch 1 \
  --global-batch 2 --max-steps 9 --stop-after-step 2 \
  --multi-node --runtime-audit-offline --runtime-audit save2
```

The timeout must fit the site's measured initialization/communication and leave
time to preserve evidence and stop paid resources. The accepted slow-link phase
took about 638 seconds. A 900-second bound is not a completion guarantee on a
different network. The launcher permits at most 1,800 seconds and no automatic
worker restarts. It kills its process group on timeout, but the operator must
also check peer processes and release the allocations/instances.

Both ranks must report finite GT forwards, actual nonzero optimizer changes,
correct global/local rank mapping and distinct physical placement. Require all
processes to exit, preserve the checkpoint, and audit its full CPU state. Check
the stage is `CHECKPOINTED`; it is not natural training completion or permission
to archive a completed run.

## Resume with new processes

After copying and independently verifying the exact `train/step-2.pt` bytes to
the same path on both hosts, start a fresh bounded phase on each host:

```bash
.venv/bin/python tools/launch_multinode.py \
  --nodes 2 --gpus-per-node 1 --node-rank "$NODE_RANK" \
  --machine-id "$THIS_MACHINE_ID" --machine-ids "$MACHINE_IDS_JSON" \
  --master-address "$MASTER_ADDRESS" --master-port 29502 \
  --cpu-input-identity "$CPU_INPUT_IDENTITY" \
  --timeout 900 --output "$SAM_WORK_ROOT/resume3" --phase train -- \
  src/train_delivery.py --config configs/engineering-public.json \
  --data-root "$SAM_WORK_ROOT/bundle" --model "$SAM_MODEL" \
  --output "$SAM_WORK_ROOT/train" --epochs 2 --microbatch 1 \
  --global-batch 2 --max-steps 9 --stop-after-step 3 \
  --resume "$SAM_WORK_ROOT/train/step-2.pt" \
  --multi-node --runtime-audit-offline --runtime-audit resume3
```

Require both ranks to restore step 2 and perform a real 2→3 optimizer update.
Audit the new checkpoint, compare its saved tensors with step 2, and confirm the
original checkpoint bytes are unchanged. Keep the same data, base, config,
world size, accumulation and per-rank RNG identity. Do not edit checkpoint
contents or replace complete resume with weights-only loading.

Each phase creates an exclusive intent. Never remove an intent or start a
duplicate because output is temporarily missing. First reconcile the processes,
results and checkpoint on both hosts. Once evidence is secured, stop the paid
instances/allocations and verify the provider's billing state; stopping the
training process alone does not stop instance charges.

## Larger layouts

The experimental parser accepts 2–16 nodes and 1–8 GPUs per node with uniform
local counts. These are software bounds. Only the 2×1 shape above was exercised
on real GPUs. Scaling changes global batch, sampler/validation boundaries,
host memory and communication cost. H200, 12-GPU layouts, full validation across
nodes and fast-network throughput require a separate bounded test on the actual
target infrastructure. The entry does not infer those settings from total GPU
count, and it does not claim elastic world-size resume.
