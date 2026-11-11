#!/bin/bash
# One arm, one torchrun. Match --gres to ABL_GPUS (default 4).
# Usage: sbatch --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=24:00:00 \
#        --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
#        --export=ALL,ABL_ARM=baseline,ABL_OUTPUT=/new/output/dir,ABL_PACKAGE=/path/to/repo,ABL_PYTHON=/path/to/env/bin/python \
#        scripts/ablation_train.sbatch
#   ABL_ARM: baseline | dual_t0 | contrastive | dual_t0_contrastive
#   Continue an unfinished arm in a new allocation: add ABL_RESUME=<run>/checkpoints/step-XXXXXXXXXXXX
#SBATCH --job-name=gs-qwen-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --output=gs-qwen-ablation-%j.log
set -euo pipefail
: "${ABL_ARM:?Set ABL_ARM}"; : "${ABL_OUTPUT:?Set ABL_OUTPUT to a new directory}"
: "${ABL_PACKAGE:?Set ABL_PACKAGE}"; : "${ABL_PYTHON:?Set ABL_PYTHON}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
env_root="$(dirname "$(dirname "$ABL_PYTHON")")"
if [[ -d "$env_root/media/lib" ]]; then export LD_LIBRARY_PATH="$env_root/media/lib:${LD_LIBRARY_PATH:-}"; export PATH="$env_root/media/bin:$PATH"; fi
cd "$ABL_PACKAGE"
args=(--arm "$ABL_ARM" --gpus "${ABL_GPUS:-4}" --epochs "${ABL_EPOCHS:-30}" --output "$ABL_OUTPUT")
if [[ -n "${ABL_RESUME:-}" ]]; then args+=(--resume "$ABL_RESUME"); fi
exec "$ABL_PYTHON" scripts/ablation_launch.py "${args[@]}"
