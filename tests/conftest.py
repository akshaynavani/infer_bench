import pytest

from infer_bench.server.config import ServerSettings, ServingMode


def make_settings(**overrides) -> ServerSettings:
    base = dict(
        step_overhead_ms=0.5,
        per_seq_cost_ms=0.05,
        prefill_cost_per_token_ms=0.0,
        cpu_bound_fraction=0.0,
        batch_wait_ms=2.0,
        max_batch_size=8,
        max_new_tokens_limit=128,
    )
    base.update(overrides)
    return ServerSettings(_env_file=None, **base)


@pytest.fixture
def fast_settings() -> ServerSettings:
    return make_settings()


@pytest.fixture
def naive_settings() -> ServerSettings:
    return make_settings(mode=ServingMode.NAIVE)
