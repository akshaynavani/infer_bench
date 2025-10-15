"""Locust entry point for infer_bench.

Run directly::

    locust -f src/infer_bench/loadgen/locustfile.py --host http://localhost:8000 \
        --workload workloads/chat_short.yaml --backend native

or via ``infer-bench run`` which wires everything up and stores results.

Each request is reported to Locust under ``<backend>:<stream|json>``. Time-to-first-token
is additionally reported as a pseudo-request of type ``LLM`` named ``ttft`` so it appears
in the Locust UI/CSV next to the end-to-end numbers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from locust import HttpUser, events, task
from locust.runners import MasterRunner, WorkerRunner

from infer_bench.loadgen.backends import get_backend
from infer_bench.loadgen.exporter import LoadgenExporter
from infer_bench.loadgen.llm_metrics import COLLECTOR, RequestSample
from infer_bench.loadgen.workload import Workload, WorkloadSampler
from infer_bench.server.model import tokenize

log = logging.getLogger("infer_bench.locust")

_STATE: dict = {"workload": None, "backend": None, "exporter": None, "user_counter": 0}


@events.init_command_line_parser.add_listener
def _add_args(parser) -> None:
    g = parser.add_argument_group("infer_bench")
    g.add_argument(
        "--backend",
        default=os.getenv("INFER_BENCH_LOADGEN_BACKEND", "native"),
        help="API adapter: native | openai | openai-chat",
    )
    g.add_argument(
        "--model",
        default=os.getenv("INFER_BENCH_MODEL_NAME", "simulated"),
        help="Model name sent to OpenAI-compatible backends",
    )
    g.add_argument(
        "--workload",
        default=os.getenv("INFER_BENCH_WORKLOAD", "workloads/throughput.yaml"),
        help="Path to a workload YAML",
    )
    g.add_argument(
        "--prometheus-port",
        type=int,
        default=int(os.getenv("INFER_BENCH_LOADGEN_PROM_PORT", "9646")),
        help="Expose load-generator metrics on this port (0 disables)",
    )
    g.add_argument(
        "--results-dir",
        default=os.getenv("INFER_BENCH_RESULTS_DIR", ""),
        help="Directory to write llm_summary.json into at test stop",
    )
    g.add_argument(
        "--api-key-env",
        default="INFER_BENCH_API_KEY",
        help="Name of the env var holding a bearer token for the target (optional)",
    )


@events.init.add_listener
def _on_init(environment, **_kwargs) -> None:
    opts = environment.parsed_options
    if opts is None:
        return
    _STATE["workload"] = Workload.load(opts.workload)
    _STATE["backend"] = get_backend(opts.backend, model=opts.model)
    if opts.prometheus_port and not isinstance(environment.runner, MasterRunner):
        exporter = LoadgenExporter()
        try:
            exporter.serve(opts.prometheus_port)
            log.info("load-generator metrics on :%d/metrics", opts.prometheus_port)
        except OSError as exc:  # port in use, e.g. several workers on one host
            log.warning("could not start Prometheus exporter: %s", exc)
        _STATE["exporter"] = exporter


@events.test_start.add_listener
def _on_test_start(environment, **_kwargs) -> None:
    COLLECTOR.reset()
    COLLECTOR.start()


@events.test_stop.add_listener
def _on_test_stop(environment, **_kwargs) -> None:
    COLLECTOR.stop()
    if isinstance(environment.runner, MasterRunner):
        return  # workers hold the samples in distributed mode
    summary = COLLECTOR.summary()
    opts = environment.parsed_options
    summary["workload"] = _STATE["workload"].name if _STATE["workload"] else None
    summary["backend"] = opts.backend if opts else None
    log.info(
        "LLM summary: %.1f req/s, %.1f out tok/s, TTFT p50 %.1f ms, p99 %.1f ms",
        summary["requests_per_s"],
        summary["output_tokens_per_s"],
        summary["ttft_ms"]["p50"],
        summary["ttft_ms"]["p99"],
    )
    results_dir = opts.results_dir if opts else ""
    if results_dir:
        suffix = ""
        if isinstance(environment.runner, WorkerRunner):
            suffix = f"_{environment.runner.client_id}"
        out = Path(results_dir) / f"llm_summary{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))


class LLMUser(HttpUser):
    """A closed-loop client issuing generation requests sampled from the workload."""

    # Overridden per instance from the workload, see wait_time().
    host = os.getenv("INFER_BENCH_TARGET", "http://localhost:8000")

    def on_start(self) -> None:
        _STATE["user_counter"] += 1
        self.sampler = WorkloadSampler(_STATE["workload"], user_index=_STATE["user_counter"])
        self.backend = _STATE["backend"]
        self.exporter: LoadgenExporter | None = _STATE["exporter"]
        api_key_env = getattr(self.environment.parsed_options, "api_key_env", "INFER_BENCH_API_KEY")
        token = os.getenv(api_key_env)
        if token:
            self.client.headers["Authorization"] = f"Bearer {token}"
        if self.exporter:
            self.exporter.users.inc()

    def on_stop(self) -> None:
        if self.exporter:
            self.exporter.users.dec()

    def wait_time(self) -> float:
        return self.sampler.wait()

    @task
    def generate(self) -> None:
        prompt, max_tokens = self.sampler.sample()
        call = self.backend.build(prompt, max_tokens, self.sampler.w.stream)
        name = f"{self.backend.name}:{'stream' if call.stream else 'json'}"

        start = time.perf_counter()
        ttft: float | None = None
        last = start
        itl: list[float] = []
        tokens = 0

        with self.client.post(
            call.path, json=call.body, stream=call.stream, catch_response=True, name=name
        ) as resp:
            try:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if call.stream:
                    for payload in self.backend.iter_sse_payloads(resp.iter_lines()):
                        n = self.backend.parse_chunk(payload)
                        if not n:
                            continue
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        else:
                            itl.append(now - last)
                        last = now
                        tokens += n
                else:
                    tokens = self.backend.parse_response(resp.json())
                    ttft = None
                if tokens == 0:
                    raise RuntimeError("response contained no tokens")
            except Exception as exc:  # noqa: BLE001 - every failure is a failed request
                resp.failure(str(exc))
                COLLECTOR.record_error()
                if self.exporter:
                    self.exporter.observe(self.backend.name, None, ok=False)
                return
            resp.success()

        sample = RequestSample(
            ttft_s=ttft,
            latency_s=time.perf_counter() - start,
            prompt_tokens=len(tokenize(prompt)),
            output_tokens=tokens,
            itl_s=itl,
        )
        COLLECTOR.record(sample)
        if self.exporter:
            self.exporter.observe(self.backend.name, sample)
        if ttft is not None:
            self.environment.events.request.fire(
                request_type="LLM",
                name="ttft",
                response_time=ttft * 1000,
                response_length=tokens,
                exception=None,
                context={},
            )
