"""Workload definitions: how prompts and output lengths are sampled.

A workload YAML looks like::

    name: chat-short
    stream: true
    seed: 7
    wait_time: {min: 0.0, max: 0.5}
    prompt_tokens: {dist: lognormal, mean: 128, sigma: 0.6, min: 8, max: 2048}
    max_new_tokens: {dist: uniform, min: 32, max: 128}
    prompts_file: null   # optional newline-separated corpus to draw prompts from
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..server.model import _WORDS


@dataclass
class LengthDist:
    dist: str = "fixed"
    value: int = 64
    min: int = 1
    max: int = 4096
    mean: float = 64.0
    sigma: float = 0.5
    std: float = 16.0

    def sample(self, rng: random.Random) -> int:
        if self.dist == "fixed":
            x = float(self.value)
        elif self.dist == "uniform":
            x = rng.uniform(self.min, self.max)
        elif self.dist == "normal":
            x = rng.gauss(self.mean, self.std)
        elif self.dist == "lognormal":
            # Parameterised by the desired mean of the distribution itself.
            mu = math.log(max(self.mean, 1.0)) - self.sigma**2 / 2
            x = rng.lognormvariate(mu, self.sigma)
        else:
            raise ValueError(f"unknown distribution '{self.dist}'")
        return int(min(max(round(x), self.min), self.max))

    @classmethod
    def from_obj(cls, obj) -> "LengthDist":
        if isinstance(obj, int):
            return cls(dist="fixed", value=obj, min=obj, max=obj)
        return cls(**obj)


@dataclass
class Workload:
    name: str = "default"
    stream: bool = True
    seed: int | None = None
    wait_min: float = 0.0
    wait_max: float = 0.0
    prompt_tokens: LengthDist = field(default_factory=lambda: LengthDist(value=128))
    max_new_tokens: LengthDist = field(default_factory=lambda: LengthDist(value=64))
    prompts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "Workload":
        wait = d.get("wait_time", {}) or {}
        prompts: list[str] = []
        if d.get("prompts_file"):
            p = Path(d["prompts_file"])
            if base_dir and not p.is_absolute():
                p = base_dir / p
            prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return cls(
            name=d.get("name", "default"),
            stream=bool(d.get("stream", True)),
            seed=d.get("seed"),
            wait_min=float(wait.get("min", 0.0)),
            wait_max=float(wait.get("max", 0.0)),
            prompt_tokens=LengthDist.from_obj(d.get("prompt_tokens", 128)),
            max_new_tokens=LengthDist.from_obj(d.get("max_new_tokens", 64)),
            prompts=prompts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Workload":
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()) or {}, base_dir=path.parent)


class WorkloadSampler:
    """Per-user sampler; each Locust user gets its own RNG stream."""

    def __init__(self, workload: Workload, user_index: int = 0):
        self.w = workload
        seed = None if workload.seed is None else workload.seed * 100_003 + user_index
        self.rng = random.Random(seed)

    def prompt(self) -> str:
        n = self.w.prompt_tokens.sample(self.rng)
        if self.w.prompts:
            base = self.rng.choice(self.w.prompts).split()
            # Repeat / truncate corpus prompt to hit the sampled length.
            words = (base * (n // max(len(base), 1) + 1))[:n]
        else:
            words = [self.rng.choice(_WORDS) for _ in range(n)]
        return " ".join(words)

    def sample(self) -> tuple[str, int]:
        return self.prompt(), self.w.max_new_tokens.sample(self.rng)

    def wait(self) -> float:
        if self.w.wait_max <= 0:
            return 0.0
        return self.rng.uniform(self.w.wait_min, self.w.wait_max)
