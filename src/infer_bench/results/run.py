"""Run directory layout.

results/
  20260108-142233_batched/
    config.json            # everything needed to reproduce the run
    locust_stats.csv       # Locust CSV outputs (prefix "locust")
    locust_stats_history.csv
    locust_failures.csv
    locust_report.html
    llm_summary.json       # client-side LLM metrics from the locustfile
    profile/               # py-spy outputs
    summary.json           # merged summary consumed by `infer-bench compare`
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


@dataclass
class RunConfig:
    label: str
    target: str
    backend: str
    workload: str
    users: int
    spawn_rate: float
    duration_s: int
    server_mode: str | None = None
    profile: bool = False
    extra: dict = field(default_factory=dict)


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def create(cls, root: str | Path, label: str) -> "RunDir":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path(root) / f"{stamp}_{_slug(label)}"
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    @property
    def locust_prefix(self) -> Path:
        return self.path / "locust"

    @property
    def locust_stats_csv(self) -> Path:
        return self.path / "locust_stats.csv"

    @property
    def locust_html(self) -> Path:
        return self.path / "locust_report.html"

    @property
    def llm_summary(self) -> Path:
        return self.path / "llm_summary.json"

    @property
    def profile_dir(self) -> Path:
        return self.path / "profile"

    @property
    def summary(self) -> Path:
        return self.path / "summary.json"

    @property
    def config(self) -> Path:
        return self.path / "config.json"

    def write_config(self, cfg: RunConfig) -> None:
        self.config.write_text(json.dumps(asdict(cfg), indent=2))

    def read_json(self, path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}
