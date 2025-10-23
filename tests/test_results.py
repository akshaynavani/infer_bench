import json

from infer_bench.profiling.pyspy import hotspots, parse_folded
from infer_bench.results.compare import compare_headlines, to_markdown
from infer_bench.results.run import RunConfig, RunDir
from infer_bench.results.summary import build_summary, load_summary, parse_locust_stats

FOLDED = """\
<module> (app.py:1);handle (app.py:10);_burn_cpu (model.py:30) 70
<module> (app.py:1);handle (app.py:10);_burn_cpu (model.py:31) 10
<module> (app.py:1);handle (app.py:12);json_dumps (json.py:5) 20
"""

LOCUST_CSV = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
    "POST,native:stream,100,2,500,520,100,900,0,10.5,0.2,500,550,600,620,700,750,800,850,890,900,900\n"
    ",Aggregated,100,2,500,520,100,900,0,10.5,0.2,500,550,600,620,700,750,800,850,890,900,900\n"
)


def test_hotspots_aggregate_lines_of_same_function():
    hs = hotspots(parse_folded(FOLDED))
    assert hs["total_samples"] == 100
    top = hs["self"][0]
    assert top["function"] == "_burn_cpu (model.py)"
    assert top["pct"] == 80.0
    incl = {r["function"]: r["pct"] for r in hs["inclusive"]}
    assert incl["handle (app.py)"] == 100.0


def test_parse_locust_stats(tmp_path):
    p = tmp_path / "locust_stats.csv"
    p.write_text(LOCUST_CSV)
    stats = parse_locust_stats(p)
    assert stats["native:stream"]["requests"] == 100
    assert stats["native:stream"]["p99"] == 850.0


def _fake_run(root, label, rps, ttft):
    run = RunDir.create(root, label)
    run.write_config(RunConfig(label, "http://x", "native", "w.yaml", 8, 1.0, 10))
    run.locust_stats_csv.write_text(LOCUST_CSV)
    run.llm_summary.write_text(json.dumps({
        "requests_per_s": rps, "output_tokens_per_s": rps * 64, "error_rate": 0.0,
        "latency_ms": {"p50": 100, "p99": 200}, "ttft_ms": {"p50": ttft, "p99": ttft * 2},
        "itl_ms": {"p50": 5}, "decode_tokens_per_s": {"p50": 200},
    }))
    run.profile_dir.mkdir()
    (run.profile_dir / "profile.folded").write_text(FOLDED)
    return run


def test_build_summary_and_compare(tmp_path):
    a = _fake_run(tmp_path, "naive", rps=10, ttft=300)
    b = _fake_run(tmp_path, "batched", rps=30, ttft=100)
    sa = build_summary(a)
    assert sa["headline"]["output_tokens_per_s"] == 640
    assert sa["profile"]["self"][0]["pct"] == 80.0

    sb = load_summary(b.path)
    deltas = {d.key: d for d in compare_headlines(sa, sb)}
    assert deltas["requests_per_s"].ratio == 3.0
    assert deltas["requests_per_s"].improved is True
    assert deltas["ttft_p50_ms"].improved is True

    md = to_markdown(sa, sb)
    assert "3.00x" in md
    assert "Requests/s" in md


def test_worker_summaries_are_merged(tmp_path):
    run = RunDir.create(tmp_path, "dist")
    for i in range(2):
        (run.path / f"llm_summary_w{i}.json").write_text(json.dumps({
            "requests": 10, "errors": 0, "output_tokens_total": 100, "prompt_tokens_total": 50,
            "requests_per_s": 5.0, "output_tokens_per_s": 50.0,
            "latency_ms": {"p99": 100 + i}, "ttft_ms": {"p99": 10}, "itl_ms": {"p99": 1},
        }))
    s = build_summary(run)
    assert s["llm"]["requests"] == 20
    assert s["headline"]["requests_per_s"] == 10.0
    assert s["llm"]["latency_ms"]["p99"] == 101
