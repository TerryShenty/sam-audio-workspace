#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
if command -v python3.12 >/dev/null 2>&1; then
  exec python3.12 "$root/bootstrap.py" "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$root/bootstrap.py" "$@"
fi
echo 'Python 3.12 is required to run the workflow.' >&2
exit 1
