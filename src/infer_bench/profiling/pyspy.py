"""Thin wrapper around the ``py-spy`` CLI.

``py-spy record`` attaches to a running Python process and samples its stacks without
modifying or restarting it. We run up to three samplers side by side (all in
``--nonblocking`` mode so the target is never paused):

* ``flamegraph`` -> ``profile.svg`` for eyeballing hot paths,
* ``speedscope`` -> ``profile.speedscope.json`` for https://www.speedscope.app,
* ``raw``        -> ``profile.folded`` (collapsed stacks) which we parse into a
  hotspot table that goes into the run summary and can be diffed between runs.

Note: on macOS py-spy needs root (``sudo``); on Linux containers it needs the
``SYS_PTRACE`` capability.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = {
    "flamegraph": "profile.svg",
    "speedscope": "profile.speedscope.json",
    "raw": "profile.folded",
}


class PySpyNotFound(RuntimeError):
    pass


def pyspy_binary() -> str:
    exe = shutil.which("py-spy")
    if exe is None:
        raise PySpyNotFound("py-spy not found on PATH; install with `pip install py-spy`")
    return exe


@dataclass
class ProfileSession:
    pid: int
    out_dir: Path
    procs: dict[str, subprocess.Popen] = field(default_factory=dict)

    def wait(self, timeout: float | None = None) -> dict[str, Path]:
        """Wait for all samplers to finish; return the produced files by format."""
        produced: dict[str, Path] = {}
        for fmt, proc in self.procs.items():
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
            path = self.out_dir / FORMATS[fmt]
            if path.exists():
                produced[fmt] = path
        return produced

    def stop(self) -> dict[str, Path]:
        """Interrupt samplers early; py-spy writes its output on SIGINT."""
        import signal

        for proc in self.procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        return self.wait(timeout=30)


class Profiler:
    def __init__(
        self,
        rate: int = 100,
        subprocesses: bool = True,
        native: bool = False,
        gil_only: bool = False,
        idle: bool = False,
        sudo: bool = False,
    ):
        self.rate = rate
        self.subprocesses = subprocesses
        self.native = native
        self.gil_only = gil_only
        self.idle = idle
        self.sudo = sudo

    def _cmd(self, pid: int, duration: int, fmt: str, out: Path) -> list[str]:
        cmd = [
            pyspy_binary(),
            "record",
            "--pid", str(pid),
            "--duration", str(duration),
            "--rate", str(self.rate),
            "--format", fmt,
            "--output", str(out),
            "--nonblocking",
        ]
        if self.subprocesses:
            cmd.append("--subprocesses")
        if self.native:
            cmd.append("--native")
        if self.gil_only:
            cmd.append("--gil")
        if self.idle:
            cmd.append("--idle")
        if self.sudo:
            cmd.insert(0, "sudo")
        return cmd

    def start(
        self,
        pid: int,
        duration: int,
        out_dir: str | Path,
        formats: tuple[str, ...] = ("flamegraph", "speedscope", "raw"),
    ) -> ProfileSession:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        session = ProfileSession(pid=pid, out_dir=out_dir)
        for fmt in formats:
            if fmt not in FORMATS:
                raise ValueError(f"unknown py-spy format '{fmt}'")
            log = open(out_dir / f"py-spy.{fmt}.log", "w")
            session.procs[fmt] = subprocess.Popen(
                self._cmd(pid, duration, fmt, out_dir / FORMATS[fmt]),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return session

    def record(self, pid: int, duration: int, out_dir: str | Path, **kw) -> dict[str, Path]:
        """Blocking convenience wrapper: start, wait, return produced files."""
        return self.start(pid, duration, out_dir, **kw).wait(timeout=duration + 60)

    def dump(self, pid: int) -> str:
        """Return a one-shot stack dump of every thread in ``pid``."""
        cmd = [pyspy_binary(), "dump", "--pid", str(pid)]
        if self.sudo:
            cmd.insert(0, "sudo")
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------------------
# Collapsed-stack analysis


def parse_folded(text: str) -> Counter:
    """Parse ``frame;frame;leaf count`` lines into a Counter of stack tuples."""
    stacks: Counter = Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        stacks[tuple(stack.split(";"))] += n
    return stacks


def _func_name(frame: str) -> str:
    # py-spy frames look like "function (path/to/file.py:123)"; drop line numbers so
    # samples from different lines of the same function aggregate together.
    name, sep, loc = frame.partition(" (")
    if not sep:
        return frame
    path = loc.rstrip(")").rsplit(":", 1)[0]
    return f"{name} ({path})"


def hotspots(stacks: Counter, top: int = 15) -> dict:
    """Self and inclusive sample share per function."""
    total = sum(stacks.values())
    self_c: Counter = Counter()
    incl_c: Counter = Counter()
    for stack, n in stacks.items():
        if not stack:
            continue
        funcs = [_func_name(f) for f in stack]
        self_c[funcs[-1]] += n
        for f in set(funcs):
            incl_c[f] += n

    def table(c: Counter) -> list[dict]:
        return [
            {"function": f, "samples": n, "pct": 100.0 * n / total if total else 0.0}
            for f, n in c.most_common(top)
        ]

    return {"total_samples": total, "self": table(self_c), "inclusive": table(incl_c)}


def hotspots_from_file(path: str | Path, top: int = 15) -> dict:
    return hotspots(parse_folded(Path(path).read_text()), top=top)
