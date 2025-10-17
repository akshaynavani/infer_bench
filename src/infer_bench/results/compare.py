"""Compare two runs: throughput / latency deltas and the profile hotspots that moved."""

from __future__ import annotations

from dataclasses import dataclass

# metric key -> (label, higher_is_better)
HEADLINE_METRICS: dict[str, tuple[str, bool]] = {
    "requests_per_s": ("Requests/s", True),
    "output_tokens_per_s": ("Output tokens/s", True),
    "decode_tps_p50": ("Per-request decode tok/s (p50)", True),
    "latency_p50_ms": ("Latency p50 (ms)", False),
    "latency_p99_ms": ("Latency p99 (ms)", False),
    "ttft_p50_ms": ("TTFT p50 (ms)", False),
    "ttft_p99_ms": ("TTFT p99 (ms)", False),
    "itl_p50_ms": ("Inter-token p50 (ms)", False),
    "error_rate": ("Error rate", False),
}


@dataclass
class MetricDelta:
    key: str
    label: str
    baseline: float
    candidate: float
    higher_is_better: bool

    @property
    def ratio(self) -> float | None:
        if self.baseline == 0:
            return None
        return self.candidate / self.baseline

    @property
    def delta_pct(self) -> float | None:
        r = self.ratio
        return None if r is None else (r - 1.0) * 100.0

    @property
    def improved(self) -> bool | None:
        if self.candidate == self.baseline:
            return None
        better = self.candidate > self.baseline
        return better if self.higher_is_better else not better


def compare_headlines(baseline: dict, candidate: dict) -> list[MetricDelta]:
    b, c = baseline.get("headline", {}), candidate.get("headline", {})
    return [
        MetricDelta(k, label, float(b.get(k, 0.0)), float(c.get(k, 0.0)), hib)
        for k, (label, hib) in HEADLINE_METRICS.items()
    ]


def compare_hotspots(baseline: dict, candidate: dict, top: int = 10) -> list[dict]:
    """Functions whose self-time share changed the most between runs."""
    def shares(s: dict) -> dict[str, float]:
        return {r["function"]: r["pct"] for r in s.get("profile", {}).get("self", [])}

    b, c = shares(baseline), shares(candidate)
    rows = [
        {"function": f, "baseline_pct": b.get(f, 0.0), "candidate_pct": c.get(f, 0.0),
         "delta_pp": c.get(f, 0.0) - b.get(f, 0.0)}
        for f in set(b) | set(c)
    ]
    rows.sort(key=lambda r: abs(r["delta_pp"]), reverse=True)
    return rows[:top]


def _fmt(x: float) -> str:
    return f"{x:,.3f}" if abs(x) < 1 else f"{x:,.1f}"


def to_markdown(baseline: dict, candidate: dict) -> str:
    lines = [
        f"## {baseline.get('run', 'baseline')} -> {candidate.get('run', 'candidate')}",
        "",
        "| Metric | Baseline | Candidate | Change |",
        "|---|---:|---:|---:|",
    ]
    for d in compare_headlines(baseline, candidate):
        if d.ratio is None:
            change = "n/a"
        else:
            arrow = {True: " (better)", False: " (worse)", None: ""}[d.improved]
            change = f"{d.ratio:.2f}x / {d.delta_pct:+.1f}%{arrow}"
        lines.append(f"| {d.label} | {_fmt(d.baseline)} | {_fmt(d.candidate)} | {change} |")

    hs = compare_hotspots(baseline, candidate)
    if hs:
        lines += ["", "### Profile hotspots (self time %)", "",
                  "| Function | Baseline | Candidate | Delta (pp) |", "|---|---:|---:|---:|"]
        for r in hs:
            lines.append(
                f"| `{r['function']}` | {r['baseline_pct']:.1f} | {r['candidate_pct']:.1f} | "
                f"{r['delta_pp']:+.1f} |"
            )
    return "\n".join(lines) + "\n"


def render_rich(baseline: dict, candidate: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"{baseline.get('run')}  vs  {candidate.get('run')}")
    for col in ("Metric", "Baseline", "Candidate", "Ratio", "Change"):
        table.add_column(col, justify="left" if col == "Metric" else "right")
    for d in compare_headlines(baseline, candidate):
        style = {True: "green", False: "red", None: ""}[d.improved]
        ratio = "n/a" if d.ratio is None else f"{d.ratio:.2f}x"
        pct = "n/a" if d.delta_pct is None else f"{d.delta_pct:+.1f}%"
        table.add_row(d.label, _fmt(d.baseline), _fmt(d.candidate), ratio, pct, style=style)
    Console().print(table)
