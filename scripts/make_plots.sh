#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v python >/dev/null 2>&1; then
  python experiments/run_all.py --plots-only
elif command -v powershell.exe >/dev/null 2>&1; then
  winpwd="$(wslpath -w "$PWD")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '$winpwd'; python experiments/run_all.py --plots-only"
else
  echo "python not found" >&2
  exit 1
fi
