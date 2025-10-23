from infer_bench.loadgen.llm_metrics import (
    LLMMetricsCollector,
    RequestSample,
    describe,
    percentile,
)


def test_percentile_interpolates():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 50) == 3
    assert percentile(xs, 100) == 5
    assert percentile([1, 2], 50) == 1.5
    assert percentile([], 50) == 0.0


def test_describe_empty():
    assert describe([])["count"] == 0


def test_decode_tps():
    s = RequestSample(ttft_s=0.1, latency_s=1.1, prompt_tokens=10, output_tokens=11)
    assert abs(s.decode_tps - 10.0) < 1e-9
    assert RequestSample(None, 1.0, 1, 5).decode_tps == 0.0


def test_collector_summary():
    c = LLMMetricsCollector()
    c.start()
    c.started_at -= 10  # pretend 10 s elapsed
    for _ in range(4):
        c.record(RequestSample(0.05, 0.5, 20, 50, itl_s=[0.01] * 49))
    c.record_error()
    c.stop()
    s = c.summary()
    assert s["requests"] == 4 and s["errors"] == 1
    assert abs(s["error_rate"] - 0.2) < 1e-9
    assert 19 < s["output_tokens_per_s"] < 21
    assert abs(s["ttft_ms"]["p50"] - 50.0) < 1e-6
    assert abs(s["itl_ms"]["p99"] - 10.0) < 1e-6
