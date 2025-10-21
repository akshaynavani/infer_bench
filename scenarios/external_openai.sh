#!/usr/bin/env bash
# Benchmark any OpenAI-compatible server (vLLM, TGI, llama.cpp, ...).
#
#   TARGET=http://gpu-box:8000 MODEL=meta-llama/Llama-3.1-8B-Instruct ./scenarios/external_openai.sh
#
# Set INFER_BENCH_API_KEY if the endpoint needs a bearer token.
# Add SERVER_PID=<pid> PROFILE=1 to also py-spy the server when it runs on this host.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TARGET:?set TARGET to the server base URL}"
MODEL="${MODEL:-default}"
BACKEND="${BACKEND:-openai}"

extra=()
if [[ "${PROFILE:-0}" == "1" ]]; then
  extra+=(--profile --server-pid "${SERVER_PID:?SERVER_PID required for PROFILE=1}")
fi

for users in ${USER_STEPS:-1 8 32 64}; do
  infer-bench run \
    --target "$TARGET" --backend "$BACKEND" --model "$MODEL" \
    --workload "${WORKLOAD:-workloads/chat_short.yaml}" \
    --users "$users" --spawn-rate "$users" --duration "${DURATION:-60}" \
    --label "${BACKEND}-u${users}" --results-root "${RESULTS:-results/external}" \
    "${extra[@]}"
done
