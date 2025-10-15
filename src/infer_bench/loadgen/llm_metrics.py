"""Client-side LLM metrics collected by the Locust process.

Locust natively tracks request latency and RPS; this module adds the numbers that
matter for generative inference: time-to-first-token, inter-token latency,
per-request decode speed, and aggregate output-token throughput.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100]); 0.0 for empty input."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


@dataclass
class RequestSample:
    ttft_s: float | None
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    itl_s: list[float] = field(default_factory=list)

    @property
    def decode_tps(self) -> float:
        """Tokens/sec during the decode phase (after the first token)."""
        if self.output_tokens <= 1 or self.ttft_s is None:
            return 0.0
        decode_time = self.latency_s - self.ttft_s
        return (self.output_tokens - 1) / decode_time if decode_time > 0 else 0.0


class LLMMetricsCollector:
    """Thread-safe accumulator of per-request samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.samples: list[RequestSample] = []
            self.errors = 0
            self.started_at: float | None = None
            self.stopped_at: float | None = None

    def start(self) -> None:
        self.started_at = time.time()

    def stop(self) -> None:
        self.stopped_at = time.time()

    def record(self, sample: RequestSample) -> None:
        with self._lock:
            self.samples.append(sample)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def summary(self) -> dict:
        with self._lock:
            samples = list(self.samples)
            errors = self.errors
        end = self.stopped_at or time.time()
        duration = (end - self.started_at) if self.started_at else 0.0
        total_out = sum(s.output_tokens for s in samples)
        total_in = sum(s.prompt_tokens for s in samples)
        itl = [x for s in samples for x in s.itl_s]
        return {
            "duration_s": duration,
            "requests": len(samples),
            "errors": errors,
            "error_rate": errors / (len(samples) + errors) if (samples or errors) else 0.0,
            "requests_per_s": len(samples) / duration if duration > 0 else 0.0,
            "output_tokens_total": total_out,
            "prompt_tokens_total": total_in,
            "output_tokens_per_s": total_out / duration if duration > 0 else 0.0,
            "latency_ms": _scale(describe([s.latency_s for s in samples]), 1000),
            "ttft_ms": _scale(describe([s.ttft_s for s in samples if s.ttft_s is not None]), 1000),
            "itl_ms": _scale(describe(itl), 1000),
            "decode_tokens_per_s": describe([s.decode_tps for s in samples if s.decode_tps > 0]),
        }


def _scale(d: dict, factor: float) -> dict:
    return {k: (v * factor if k != "count" else v) for k, v in d.items()}


# Process-wide collector used by the locustfile.
COLLECTOR = LLMMetricsCollector()
