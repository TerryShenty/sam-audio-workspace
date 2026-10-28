#!/bin/bash
#SBATCH --job-name=qwen30b-sam-v6-auto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=qwen30b-%j.log
set -euo pipefail
: "${QWEN_PACKAGE:?Set QWEN_PACKAGE to the repository directory}"
: "${QWEN_PYTHON:?Set QWEN_PYTHON to your environment Python}"
: "${QWEN_OUTPUT:?Set QWEN_OUTPUT to a new output directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
env_root="$(dirname "$(dirname "$QWEN_PYTHON")")"
if [[ -d "$env_root/media/lib" ]]; then
 export LD_LIBRARY_PATH="$env_root/media/lib:${LD_LIBRARY_PATH:-}"
 export PATH="$env_root/media/bin:$PATH"
fi
cd "$QWEN_PACKAGE"
args=(--mode "${QWEN_MODE:-train}" --output "$QWEN_OUTPUT")
if [[ -n "${QWEN_GATE:-}" ]]; then args+=(--gate "$QWEN_GATE"); fi
if [[ -n "${QWEN_RESUME:-}" ]]; then args+=(--resume "$QWEN_RESUME"); fi
exec "$QWEN_PYTHON" scripts/launch.py "${args[@]}"
