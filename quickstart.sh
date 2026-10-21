#!/usr/bin/env bash
# A small, pinned entry to the accepted engineering workflow.
set -euo pipefail
PY="${PYTHON_BIN:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo 'Install Python 3.12, or set PYTHON_BIN to its executable.' >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo 'curl is required to fetch the verified bootstrap.' >&2
  exit 2
fi
echo 'SAM Audio: 3-video / 13-prompt engineering check + full285 validation; not full93942 training.' >&2
tmp="$(mktemp -d "${TMPDIR:-/tmp}/sam-quickstart.XXXXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT
curl --fail --silent --show-error --location --connect-timeout 20 --max-time 180 \
  'https://raw.githubusercontent.com/TerryShenty/sam-audio-workspace/56e7db0a52e0bf539b4f5ee3f365adb797e16273/bootstrap.py' \
  --output "$tmp/bootstrap.py"
"$PY" - "$tmp/bootstrap.py" <<'PY'
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
if hashlib.sha256(path.read_bytes()).hexdigest() != 'e4c2ff4bdffccc4db0fc7738e5384fe8a4d9c5a18fcbcb27de803eb6cb2cd096':
    raise SystemExit('Bootstrap SHA256 mismatch; nothing was executed.')
PY
# Later user arguments override these modest engineering defaults.
# Omitted global batch is derived from microbatch * selected GPU count.
"$PY" "$tmp/bootstrap.py" --directory "${SAM_RUN_DIR:-$PWD/sam-run}" \
  --gpus 2 --performance-profile solos-fast-v1 --microbatch 2 "$@"
