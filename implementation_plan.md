# infer_bench — Implementation Plan

**Load & Profiling Harness for LLM Inference**

Tech: Python, Locust, py-spy, Prometheus, Grafana, FastAPI

## Goal

A Python benchmarking framework that profiles and evaluates LLM inference backends under load:

- **Load generation** with Locust against any inference backend (the bundled FastAPI server,
  or any OpenAI-compatible endpoint such as vLLM / TGI / llama.cpp server).
- **LLM-aware metrics**: time-to-first-token (TTFT), inter-token latency, tokens/sec,
  end-to-end latency percentiles, error rate.
- **CPU profiling** of the server process with py-spy (flamegraphs + speedscope output)
  captured while the load test runs.
- **Real-time dashboards** with Prometheus + Grafana (server-side and load-generator-side metrics).
- A **reference FastAPI inference server** with two serving modes — a naive one
  (blocking, one request at a time per worker) and an optimized one (async + dynamic
  micro-batching) — so the harness can demonstrate and quantify a throughput improvement
  (target: ~3x) between configurations.

## Key design decisions

- **Backend-agnostic adapters.** A small `Backend` abstraction maps a generic "generate"
  request onto a concrete HTTP API (native `infer_bench` server, OpenAI-compatible
  `/v1/completions`). Locust users call through the adapter, so swapping targets is a flag.
- **Mock model by default.** The reference server ships a deterministic simulated model
  (CPU-bound per-token work + configurable per-step latency) so everything runs on a laptop
  with no GPU or weights. An optional HuggingFace `transformers` backend is pluggable but
  not required.
- **Metrics in two places.** The server exports Prometheus metrics via `prometheus_client`;
  the Locust process exports its own LLM metrics on a separate port, so Grafana can show
  both sides in one dashboard.
- **Run artifacts on disk.** Each run writes `results/<run_id>/` with config, Locust CSV
  stats, a JSON summary, and py-spy profiles. A `compare` command diffs two runs.
- **Single CLI** (`infer-bench`) built with Typer orchestrates server, profiler and Locust.

## Phases

### Phase 1 — Project scaffold & reference inference server
Deliverables:
- `pyproject.toml` with package metadata, deps, and `infer-bench` console script.
- Config models (pydantic settings) for the server.
- Simulated model (`SimulatedLLM`) with tokenizer stub, CPU-bound per-token work, streaming.
- FastAPI app with `/health`, `/v1/generate` (JSON + SSE streaming) in two modes:
  `naive` (sync, per-request) and `batched` (async queue + dynamic micro-batcher).
- Prometheus instrumentation on the server (`/metrics`): request count, latency histogram,
  TTFT histogram, tokens generated, batch size histogram, queue depth gauge.

Files: `pyproject.toml`, `src/infer_bench/__init__.py`, `src/infer_bench/server/{config,model,batcher,metrics,app}.py`

### Phase 2 — Load generation with Locust
Deliverables:
- Backend adapters (`native`, `openai`) with streaming parsing and TTFT capture.
- Workload definitions: prompt length / max-token distributions, loaded from YAML.
- `locustfile.py` with an `LLMUser` that fires custom Locust events (TTFT, tokens/s).
- Locust-side Prometheus exporter for LLM metrics.
- Example workload YAMLs.

Files: `src/infer_bench/loadgen/{backends,workload,llm_metrics,exporter,locustfile}.py`, `workloads/*.yaml`

### Phase 3 — Profiling with py-spy & run results
Deliverables:
- `Profiler` wrapper around `py-spy record` (flamegraph SVG + speedscope JSON) and
  `py-spy dump`, attaching to a PID for a given duration.
- Run directory management and JSON summary built from Locust CSV stats + LLM metrics.
- `compare` logic producing a markdown/console table of throughput and latency deltas.

Files: `src/infer_bench/profiling/pyspy.py`, `src/infer_bench/results/{run,summary,compare}.py`

### Phase 4 — Observability stack (Prometheus + Grafana)
Deliverables:
- `docker-compose.yml` with server, Prometheus, Grafana.
- Prometheus scrape config for server and Locust exporter.
- Grafana provisioning (datasource + dashboard JSON) with throughput, latency percentiles,
  TTFT, batch size, queue depth, and tokens/sec panels.
- Dockerfile for the server.

Files: `docker-compose.yml`, `Dockerfile`, `deploy/prometheus/prometheus.yml`,
`deploy/grafana/provisioning/**`, `deploy/grafana/dashboards/infer_bench.json`

### Phase 5 — CLI orchestration, scenarios & tests
Deliverables:
- Typer CLI: `serve`, `run` (start server optional → profiler → headless Locust → summary),
  `profile`, `compare`.
- `scenarios/naive_vs_batched.sh` script reproducing the throughput comparison.
- pytest suite: simulated model, batcher, API endpoints, workload sampling, summary/compare.
- `.env.example`, `.gitignore`, `README.md` with architecture diagram and run/test docs.

Files: `src/infer_bench/cli.py`, `scenarios/*`, `tests/*`, `.env.example`, `.gitignore`, `README.md`
