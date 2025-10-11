"""Prometheus exporter for load-generator-side LLM metrics.

Started from the locustfile on ``--prometheus-port`` (default 9646) so Grafana can
plot what clients actually observe next to what the server reports.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from ..server.metrics import LATENCY_BUCKETS, TTFT_BUCKETS


class LoadgenExporter:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "loadgen_requests_total", "Requests issued", ["backend", "status"], registry=r
        )
        self.latency = Histogram(
            "loadgen_request_latency_seconds",
            "Client-observed end-to-end latency",
            ["backend"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "loadgen_ttft_seconds",
            "Client-observed time to first token",
            ["backend"],
            buckets=TTFT_BUCKETS,
            registry=r,
        )
        self.itl = Histogram(
            "loadgen_inter_token_latency_seconds",
            "Gap between consecutive streamed tokens",
            ["backend"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=r,
        )
        self.output_tokens = Counter(
            "loadgen_output_tokens_total", "Output tokens received", ["backend"], registry=r
        )
        self.users = Gauge("loadgen_active_users", "Simulated users running", registry=r)

    def serve(self, port: int, addr: str = "0.0.0.0") -> None:
        start_http_server(port, addr=addr, registry=self.registry)

    def observe(self, backend: str, sample, ok: bool = True) -> None:
        self.requests.labels(backend, "ok" if ok else "error").inc()
        if not ok or sample is None:
            return
        self.latency.labels(backend).observe(sample.latency_s)
        if sample.ttft_s is not None:
            self.ttft.labels(backend).observe(sample.ttft_s)
        for gap in sample.itl_s:
            self.itl.labels(backend).observe(gap)
        self.output_tokens.labels(backend).inc(sample.output_tokens)
