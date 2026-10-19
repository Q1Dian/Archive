#!/usr/bin/env bash
# Run a command in its own memory-capped systemd user scope, detached, logging to a file.
# If the job exceeds its cap, only the job is killed, not the desktop or the editor.
#
#   scripts/run_capped.sh <mem cap, e.g. 20G> <log file> <command...>
# Keep the sum of all job caps well below physical RAM (32 GB here).
set -euo pipefail
cap=$1; log=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$log")"
setsid nohup systemd-run --user --scope --quiet \
    -p MemoryMax="$cap" -p MemorySwapMax=512M \
    nice -n 5 "$@" >> "$log" 2>&1 < /dev/null &
echo "started (cap $cap): $* -> $log"
