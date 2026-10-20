#!/usr/bin/env bash
# One entry for environment preparation. Never starts a long training run.
set -euo pipefail
if [[ "${1:-}" == "--help" || $# -ne 1 ]]; then
  echo 'Usage: bash start.sh /absolute/path/to/workspace'
  echo 'Prepares MSDWild78 + EasyCom207 validation media and an isolated environment.'
  echo 'Requires Linux x86_64, Python 3.12, git, FFmpeg, curl and outbound HTTPS.'
  echo 'Preparation only: full training delivery has not passed acceptance yet.'
  exit 0
fi
case "$1" in /*) ;; *) echo 'Choose an absolute workspace path.' >&2; exit 2;; esac
TASK_DIR="$1"
for command_name in curl git ffmpeg; do
  command -v "$command_name" >/dev/null || { echo "Server prerequisite missing: $command_name. Ask its owner to install it once." >&2; exit 2; }
done
command -v python3.12 >/dev/null || { echo "Server prerequisite missing: python3.12" >&2; exit 2; }
TASK_PYTHON=python3.12
"$TASK_PYTHON" -c 'import sys,platform; assert sys.version_info[:2]==(3,12), "Python 3.12 is required"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Linux x86_64 is required"' || { echo 'Host prerequisites incomplete; no environment changed.' >&2; exit 2; }
mkdir -p "$TASK_DIR"
TASK_LOG="$TASK_DIR/preparation-console.log"
TASK_BOOTSTRAP=$(mktemp)
trap 'rm -f "$TASK_BOOTSTRAP"' EXIT
curl -fL --retry 3 https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/main/bootstrap.py -o "$TASK_BOOTSTRAP"
set +e
"$TASK_PYTHON" "$TASK_BOOTSTRAP" --directory "$TASK_DIR" --prepare-validation --install 2>&1 | tee -a "$TASK_LOG"
TASK_RC=${PIPESTATUS[0]}
set -e
if (( TASK_RC != 0 )); then
  echo "Preparation stopped. Logs: $TASK_LOG and $TASK_DIR/installation.log"
  echo 'Retry this same command after the reported issue is fixed. No training was launched.'
  exit "$TASK_RC"
fi
printf '%s\n' 'Validation media and environment prepared. GPU feature caches and training acceptance are still required.'
