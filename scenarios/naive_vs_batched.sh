#!/usr/bin/env bash
# Reproduce the naive-vs-batched throughput comparison on the reference server.
#
#   ./scenarios/naive_vs_batched.sh            # 60 s per mode, 32 users
#   DURATION=120 USERS=64 PROFILE=1 ./scenarios/naive_vs_batched.sh
#
# PROFILE=1 attaches py-spy to the server (add SUDO_PYSPY=1 on macOS).
set -euo pipefail

cd "$(dirname "$0")/.."

DURATION="${DURATION:-60}"
USERS="${USERS:-32}"
SPAWN_RATE="${SPAWN_RATE:-8}"
WORKLOAD="${WORKLOAD:-workloads/throughput.yaml}"
PORT="${PORT:-8100}"
RESULTS="${RESULTS:-results/naive_vs_batched}"

extra=()
if [[ "${PROFILE:-0}" == "1" ]]; then
  extra+=(--profile)
  [[ "${SUDO_PYSPY:-0}" == "1" ]] && extra+=(--sudo-pyspy)
fi

for mode in naive batched; do
  infer-bench run \
    --spawn-server --mode "$mode" \
    --target "http://127.0.0.1:${PORT}" \
    --workload "$WORKLOAD" \
    --users "$USERS" --spawn-rate "$SPAWN_RATE" --duration "$DURATION" \
    --label "$mode" --results-root "$RESULTS" \
    "${extra[@]}"
done

naive_dir=$(ls -d "$RESULTS"/*_naive | tail -n 1)
batched_dir=$(ls -d "$RESULTS"/*_batched | tail -n 1)
infer-bench compare "$naive_dir" "$batched_dir" --markdown "$RESULTS/comparison.md"
