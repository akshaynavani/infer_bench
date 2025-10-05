"""Prometheus metrics exported by the reference server on ``/metrics``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 30.0, 60.0
)
TTFT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)
BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128)


class ServerMetrics:
    """Holds all server-side metrics in a dedicated registry (keeps tests isolated)."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "infer_requests_total", "Generation requests", ["mode", "status"], registry=r
        )
        self.in_flight = Gauge("infer_requests_in_flight", "Requests in progress", registry=r)
        self.latency = Histogram(
            "infer_request_latency_seconds",
            "End-to-end request latency",
            ["mode"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "infer_time_to_first_token_seconds",
            "Time from request arrival to first generated token",
            ["mode"],
            buckets=TTFT_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "infer_prompt_tokens_total", "Prompt tokens processed", registry=r
        )
        self.generated_tokens = Counter(
            "infer_generated_tokens_total", "Tokens generated", registry=r
        )
        self.batch_size = Histogram(
            "infer_batch_size", "Sequences per decode step", buckets=BATCH_BUCKETS, registry=r
        )
        self.step_latency = Histogram(
            "infer_decode_step_seconds",
            "Wall time of one decode step",
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=r,
        )
        self.queue_depth = Gauge(
            "infer_queue_depth", "Requests waiting to be admitted to a batch", registry=r
        )
        self.active_sequences = Gauge(
            "infer_active_sequences", "Sequences currently being decoded", registry=r
        )
