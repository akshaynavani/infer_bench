"""Server configuration, read from environment variables prefixed with INFER_BENCH_."""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingMode(str, Enum):
    """How the server schedules generation work.

    naive:   every request runs the model synchronously inside the request handler.
             FastAPI pushes sync handlers onto a threadpool, so requests contend on the GIL
             and each forward pass handles exactly one sequence.
    batched: requests are queued and a background batcher groups them into micro-batches
             so one forward step advances many sequences at once.
    """

    NAIVE = "naive"
    BATCHED = "batched"


class ModelBackend(str, Enum):
    SIMULATED = "simulated"
    HF = "hf"


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFER_BENCH_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    mode: ServingMode = ServingMode.BATCHED
    backend: ModelBackend = ModelBackend.SIMULATED

    # Simulated model knobs
    step_overhead_ms: float = Field(
        default=8.0, description="Fixed cost per forward step regardless of batch size"
    )
    per_seq_cost_ms: float = Field(
        default=1.0, description="Marginal cost per sequence in a forward step"
    )
    prefill_cost_per_token_ms: float = Field(
        default=0.05, description="Prompt processing cost per prompt token"
    )
    cpu_bound_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of simulated cost spent busy-looping (visible to py-spy) vs sleeping",
    )
    vocab_size: int = 32000
    seed: int = 1234

    # Batcher knobs
    max_batch_size: int = 32
    batch_wait_ms: float = 5.0
    max_queue_size: int = 1024

    # HF backend
    hf_model_name: str = "sshleifer/tiny-gpt2"

    # Request limits
    max_new_tokens_limit: int = 1024
