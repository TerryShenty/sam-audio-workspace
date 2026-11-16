#!/bin/bash
# Build one cache stage on all GPUs of this allocation, one shard per GPU, then finalize.
# Usage: sbatch --gres=gpu:N --cpus-per-task=$((4*N)) --mem=$((32*N))G --time=08:00:00 \
#        --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
#        --export=ALL,STAGE=sam|qwen,ABL_PACKAGE=/path/to/repo,ABL_PYTHON=/path/to/env/bin/python \
#        scripts/ablation_cache.sbatch
#SBATCH --job-name=ablation-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=ablation-cache-%j.log
set -euo pipefail
: "${STAGE:?Set STAGE=sam or STAGE=qwen}"
: "${ABL_PACKAGE:?Set ABL_PACKAGE to the repository directory}"
: "${ABL_PYTHON:?Set ABL_PYTHON to your environment Python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
env_root="$(dirname "$(dirname "$ABL_PYTHON")")"
if [[ -d "$env_root/media/lib" ]]; then export LD_LIBRARY_PATH="$env_root/media/lib:${LD_LIBRARY_PATH:-}"; export PATH="$env_root/media/bin:$PATH"; fi
cd "$ABL_PACKAGE"
script="scripts/build_${STAGE}_cache.py"
n=$("$ABL_PYTHON" -c 'import torch;print(torch.cuda.device_count())')
pids=()
for ((i=0; i<n; i++)); do
  CUDA_VISIBLE_DEVICES=$i "$ABL_PYTHON" "$script" --shard "$i" --num-shards "$n" > "cache-${STAGE}-shard${i}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ $status -ne 0 ]]; then echo "A shard failed; see cache-${STAGE}-shard*-${SLURM_JOB_ID}.log. Rerunning reuses finished entries." >&2; exit 1; fi
CUDA_VISIBLE_DEVICES= "$ABL_PYTHON" "$script" --finalize
