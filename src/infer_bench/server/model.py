"""Model backends for the reference server.

The default ``SimulatedLLM`` needs no GPU or weights. It reproduces the cost structure
that matters for serving performance:

* a prefill phase whose cost grows with prompt length,
* a decode step with a fixed per-step overhead plus a smaller marginal cost per sequence
  in the batch (this is what makes batching pay off on real accelerators), and
* a configurable split between CPU-bound work (holds the GIL, shows up in py-spy
  flamegraphs) and wait time (sleeps, like waiting on a GPU kernel).

Generated text is deterministic for a given prompt and seed so tests are stable.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence

from .config import ServerSettings

_WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at "
    "which but have an they you were her she there been one all we their has would when if "
    "so no will more about can my what up out them into only some time could these two may "
    "then do first any like now over such our man me even most made after also did many "
    "model token latency throughput batch queue kernel memory cache layer attention"
).split()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer — good enough for counting and simulating cost."""
    return text.split()


def _burn_cpu(duration_s: float) -> int:
    """Spin on hashing for ``duration_s`` seconds while holding the GIL."""
    deadline = time.perf_counter() + duration_s
    acc = 0
    payload = b"infer_bench"
    while time.perf_counter() < deadline:
        payload = hashlib.blake2b(payload, digest_size=16).digest()
        acc ^= payload[0]
    return acc


def spend(cost_ms: float, cpu_fraction: float) -> None:
    """Consume ``cost_ms`` of wall time split between CPU work and sleeping."""
    if cost_ms <= 0:
        return
    cpu_s = cost_ms * cpu_fraction / 1000.0
    wait_s = cost_ms * (1.0 - cpu_fraction) / 1000.0
    if cpu_s > 0:
        _burn_cpu(cpu_s)
    if wait_s > 0:
        time.sleep(wait_s)


@dataclass
class Sequence_:
    """State of one in-flight generation."""

    prompt_tokens: list[str]
    max_new_tokens: int
    seed: int
    generated: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


class LLMBackend(Protocol):
    def prefill(self, seqs: Sequence[Sequence_]) -> None: ...

    def decode_step(self, seqs: Sequence[Sequence_]) -> list[str]: ...

    def new_sequence(self, prompt: str, max_new_tokens: int) -> Sequence_: ...


class SimulatedLLM:
    """Deterministic, cost-modelled stand-in for a real LLM."""

    def __init__(self, settings: ServerSettings):
        self.s = settings

    def new_sequence(self, prompt: str, max_new_tokens: int) -> Sequence_:
        digest = hashlib.sha256(f"{self.s.seed}:{prompt}".encode()).digest()
        return Sequence_(
            prompt_tokens=tokenize(prompt),
            max_new_tokens=max_new_tokens,
            seed=int.from_bytes(digest[:8], "little"),
        )

    def prefill(self, seqs: Sequence[Sequence_]) -> None:
        n_tokens = sum(len(s.prompt_tokens) for s in seqs)
        cost = self.s.step_overhead_ms + n_tokens * self.s.prefill_cost_per_token_ms
        spend(cost, self.s.cpu_bound_fraction)

    def decode_step(self, seqs: Sequence[Sequence_]) -> list[str]:
        active = [s for s in seqs if not s.done]
        if not active:
            return []
        cost = self.s.step_overhead_ms + len(active) * self.s.per_seq_cost_ms
        spend(cost, self.s.cpu_bound_fraction)
        out = []
        for seq in active:
            tok = self._next_token(seq)
            seq.generated.append(tok)
            out.append(tok)
        return out

    @staticmethod
    def _next_token(seq: Sequence_) -> str:
        # Cheap LCG keyed on sequence seed and position -> deterministic output.
        x = (seq.seed + 6364136223846793005 * (len(seq.generated) + 1)) & 0xFFFFFFFFFFFFFFFF
        return _WORDS[x % len(_WORDS)]

    def generate_stream(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        """Single-sequence generation, used by the naive serving mode."""
        seq = self.new_sequence(prompt, max_new_tokens)
        self.prefill([seq])
        while not seq.done:
            (tok,) = self.decode_step([seq])
            yield tok


def build_backend(settings: ServerSettings) -> SimulatedLLM:
    from .config import ModelBackend

    if settings.backend == ModelBackend.SIMULATED:
        return SimulatedLLM(settings)
    if settings.backend == ModelBackend.HF:
        from .hf_backend import HFBackend

        return HFBackend(settings)  # type: ignore[return-value]
    raise ValueError(f"unknown backend {settings.backend}")
