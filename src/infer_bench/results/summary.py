"""Merge Locust CSV stats, client LLM metrics and profiler hotspots into summary.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..profiling.pyspy import FORMATS, hotspots_from_file
from .run import RunDir

_PCT_COLS = {"p50": "50%", "p90": "90%", "p95": "95%", "p99": "99%"}


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def parse_locust_stats(csv_path: Path) -> dict:
    """Return per-endpoint stats from ``locust_stats.csv`` keyed by request name."""
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            name = row.get("Name", "")
            out[name] = {
                "type": row.get("Type", ""),
                "requests": int(_f(row, "Request Count")),
                "failures": int(_f(row, "Failure Count")),
                "rps": _f(row, "Requests/s"),
                "avg_ms": _f(row, "Average Response Time"),
                "max_ms": _f(row, "Max Response Time"),
                **{k: _f(row, col) for k, col in _PCT_COLS.items()},
            }
    return out


def _merge_worker_summaries(run: RunDir) -> dict:
    """In distributed mode each worker writes llm_summary_<id>.json; combine them."""
    if run.llm_summary.exists():
        return json.loads(run.llm_summary.read_text())
    parts = [json.loads(p.read_text()) for p in sorted(run.path.glob("llm_summary_*.json"))]
    if not parts:
        return {}
    merged = dict(parts[0])
    for key in ("requests", "errors", "output_tokens_total", "prompt_tokens_total"):
        merged[key] = sum(p.get(key, 0) for p in parts)
    for key in ("requests_per_s", "output_tokens_per_s"):
        merged[key] = sum(p.get(key, 0.0) for p in parts)
    merged["workers"] = len(parts)
    # Percentiles cannot be merged exactly; keep the worst worker's tail as a bound.
    for key in ("latency_ms", "ttft_ms", "itl_ms"):
        merged[key] = max((p.get(key, {}) for p in parts), key=lambda d: d.get("p99", 0.0))
    return merged


def build_summary(run: RunDir) -> dict:
    config = run.read_json(run.config)
    llm = _merge_worker_summaries(run)
    locust = parse_locust_stats(run.locust_stats_csv)
    profile: dict = {}
    folded = run.profile_dir / FORMATS["raw"]
    if folded.exists() and folded.stat().st_size > 0:
        profile = hotspots_from_file(folded)
        profile["files"] = sorted(p.name for p in run.profile_dir.iterdir() if p.is_file())

    headline = {
        "requests_per_s": llm.get("requests_per_s", 0.0),
        "output_tokens_per_s": llm.get("output_tokens_per_s", 0.0),
        "error_rate": llm.get("error_rate", 0.0),
        "latency_p50_ms": llm.get("latency_ms", {}).get("p50", 0.0),
        "latency_p99_ms": llm.get("latency_ms", {}).get("p99", 0.0),
        "ttft_p50_ms": llm.get("ttft_ms", {}).get("p50", 0.0),
        "ttft_p99_ms": llm.get("ttft_ms", {}).get("p99", 0.0),
        "itl_p50_ms": llm.get("itl_ms", {}).get("p50", 0.0),
        "decode_tps_p50": llm.get("decode_tokens_per_s", {}).get("p50", 0.0),
    }
    summary = {
        "run": run.path.name,
        "config": config,
        "headline": headline,
        "llm": llm,
        "locust": locust,
        "profile": profile,
    }
    run.summary.write_text(json.dumps(summary, indent=2))
    return summary


def load_summary(path: str | Path) -> dict:
    """Accept either a run directory or a summary.json path."""
    p = Path(path)
    if p.is_dir():
        run = RunDir(p)
        if not run.summary.exists():
            return build_summary(run)
        p = run.summary
    return json.loads(p.read_text())
