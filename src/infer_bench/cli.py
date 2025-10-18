"""``infer-bench`` command line interface."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from .profiling.pyspy import Profiler, hotspots_from_file, FORMATS
from .results.compare import render_rich, to_markdown
from .results.run import RunConfig, RunDir
from .results.summary import build_summary, load_summary

app = typer.Typer(help="Load and profiling harness for LLM inference backends.", no_args_is_help=True)
console = Console()

LOCUSTFILE = Path(__file__).parent / "loadgen" / "locustfile.py"


# ---------------------------------------------------------------------------
# serve


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
    mode: str = typer.Option(
        os.getenv("INFER_BENCH_MODE", "batched"), help="Serving mode: naive | batched"
    ),
    backend: str = typer.Option(
        os.getenv("INFER_BENCH_BACKEND", "simulated"), help="Model backend: simulated | hf"
    ),
    max_batch_size: Optional[int] = typer.Option(None, help="Override max batch size"),
    log_level: str = typer.Option("warning", help="uvicorn log level"),
) -> None:
    """Run the reference FastAPI inference server."""
    import uvicorn

    os.environ["INFER_BENCH_MODE"] = mode
    os.environ["INFER_BENCH_BACKEND"] = backend
    if max_batch_size is not None:
        os.environ["INFER_BENCH_MAX_BATCH_SIZE"] = str(max_batch_size)
    console.print(f"[bold]infer_bench server[/] mode={mode} backend={backend} on {host}:{port}")
    uvicorn.run(
        "infer_bench.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# run


def _wait_healthy(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"server at {url} did not become healthy: {last_err}")


def _spawn_server(port: int, mode: str, log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "INFER_BENCH_MODE": mode}
    log = open(log_path, "w")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "infer_bench.server.app:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _print_headline(summary: dict) -> None:
    h = summary["headline"]
    t = Table(title=f"Run {summary['run']}")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in h.items():
        t.add_row(k, f"{v:,.3f}" if abs(v) < 1 else f"{v:,.1f}")
    console.print(t)
    prof = summary.get("profile") or {}
    if prof.get("self"):
        pt = Table(title="Top self-time functions (py-spy)")
        pt.add_column("Function")
        pt.add_column("%", justify="right")
        for row in prof["self"][:8]:
            pt.add_row(row["function"], f"{row['pct']:.1f}")
        console.print(pt)


@app.command()
def run(
    target: str = typer.Option("http://localhost:8000", help="Base URL of the backend under test"),
    backend: str = typer.Option("native", help="API adapter: native | openai | openai-chat"),
    model: str = typer.Option("simulated", help="Model name for OpenAI-compatible targets"),
    workload: Path = typer.Option(Path("workloads/throughput.yaml"), exists=True, help="Workload YAML"),
    users: int = typer.Option(32, help="Concurrent simulated users"),
    spawn_rate: float = typer.Option(8.0, help="Users started per second"),
    duration: int = typer.Option(60, help="Test duration in seconds"),
    label: str = typer.Option("run", help="Label used in the results directory name"),
    results_root: Path = typer.Option(Path("results"), help="Where run directories go"),
    spawn_server: bool = typer.Option(
        False, "--spawn-server/--no-spawn-server", help="Start the reference server for this run"
    ),
    mode: str = typer.Option("batched", help="Serving mode when --spawn-server is used"),
    profile: bool = typer.Option(False, "--profile/--no-profile", help="Profile the server with py-spy"),
    server_pid: Optional[int] = typer.Option(None, help="PID to profile when not spawning the server"),
    pyspy_rate: int = typer.Option(100, help="py-spy samples per second"),
    sudo_pyspy: bool = typer.Option(False, help="Run py-spy via sudo (needed on macOS)"),
    prometheus_port: int = typer.Option(9646, help="Load-generator metrics port (0 disables)"),
) -> None:
    """Run a headless load test, optionally spawning and profiling the server."""
    run_dir = RunDir.create(results_root, label)
    cfg = RunConfig(
        label=label,
        target=target,
        backend=backend,
        workload=str(workload),
        users=users,
        spawn_rate=spawn_rate,
        duration_s=duration,
        server_mode=mode if spawn_server else None,
        profile=profile,
        extra={"model": model, "pyspy_rate": pyspy_rate},
    )
    run_dir.write_config(cfg)
    console.print(f"[bold]run dir[/] {run_dir.path}")

    server: subprocess.Popen | None = None
    session = None
    try:
        if spawn_server:
            port = httpx.URL(target).port or 8000
            server = _spawn_server(port, mode, run_dir.path / "server.log")
            _wait_healthy(target)
            server_pid = server.pid
            console.print(f"spawned server pid={server_pid} mode={mode}")
        else:
            _wait_healthy(target, timeout=5.0)

        if profile:
            if server_pid is None:
                raise typer.BadParameter("--profile needs --spawn-server or --server-pid")
            session = Profiler(rate=pyspy_rate, sudo=sudo_pyspy).start(
                server_pid, duration, run_dir.profile_dir
            )
            console.print(f"py-spy attached to pid {server_pid}")

        locust_cmd = [
            sys.executable, "-m", "locust",
            "-f", str(LOCUSTFILE),
            "--headless",
            "--host", target,
            "-u", str(users),
            "-r", str(spawn_rate),
            "-t", f"{duration}s",
            "--stop-timeout", "30",
            "--csv", str(run_dir.locust_prefix),
            "--csv-full-history",
            "--html", str(run_dir.locust_html),
            "--only-summary",
            "--workload", str(workload),
            "--backend", backend,
            "--model", model,
            "--results-dir", str(run_dir.path),
            "--prometheus-port", str(prometheus_port),
        ]
        console.print(f"locust: {users} users for {duration}s against {target}")
        rc = subprocess.call(locust_cmd)
        if rc not in (0, 1):  # locust exits 1 when any request failed
            console.print(f"[red]locust exited with code {rc}[/]")

        if session is not None:
            produced = session.wait(timeout=60)
            console.print(f"profiles: {', '.join(p.name for p in produced.values()) or 'none'}")
    finally:
        if session is not None:
            session.stop()
        if server is not None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()

    summary = build_summary(run_dir)
    _print_headline(summary)
    console.print(f"summary written to {run_dir.summary}")


# ---------------------------------------------------------------------------
# profile / summarize / compare


@app.command("profile")
def profile_cmd(
    pid: int = typer.Option(..., help="PID of the Python process to sample"),
    duration: int = typer.Option(30, help="Seconds to sample for"),
    out: Path = typer.Option(Path("results/profile"), help="Output directory"),
    rate: int = typer.Option(100, help="Samples per second"),
    sudo: bool = typer.Option(False, help="Run py-spy via sudo (needed on macOS)"),
    native: bool = typer.Option(False, help="Include native (C/C++) frames"),
) -> None:
    """Attach py-spy to a running process and report its hotspots."""
    produced = Profiler(rate=rate, sudo=sudo, native=native).record(pid, duration, out)
    for fmt, path in produced.items():
        console.print(f"{fmt:>10}: {path}")
    folded = out / FORMATS["raw"]
    if folded.exists():
        hs = hotspots_from_file(folded)
        t = Table(title=f"Hotspots ({hs['total_samples']} samples)")
        t.add_column("Function")
        t.add_column("Self %", justify="right")
        for row in hs["self"]:
            t.add_row(row["function"], f"{row['pct']:.1f}")
        console.print(t)


@app.command()
def summarize(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """(Re)build summary.json for a run directory."""
    summary = build_summary(RunDir(run_dir))
    _print_headline(summary)


@app.command()
def compare(
    baseline: Path = typer.Argument(..., exists=True, help="Baseline run dir or summary.json"),
    candidate: Path = typer.Argument(..., exists=True, help="Candidate run dir or summary.json"),
    markdown: Optional[Path] = typer.Option(None, help="Also write a markdown report here"),
    as_json: bool = typer.Option(False, "--json", help="Print headline deltas as JSON"),
) -> None:
    """Compare two runs side by side."""
    b, c = load_summary(baseline), load_summary(candidate)
    if as_json:
        from .results.compare import compare_headlines

        rows = [
            {"metric": d.key, "baseline": d.baseline, "candidate": d.candidate, "ratio": d.ratio}
            for d in compare_headlines(b, c)
        ]
        console.print_json(json.dumps(rows))
    else:
        render_rich(b, c)
    if markdown:
        markdown.write_text(to_markdown(b, c))
        console.print(f"markdown report: {markdown}")


if __name__ == "__main__":  # pragma: no cover
    app()
