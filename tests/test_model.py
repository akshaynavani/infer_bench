import time

from infer_bench.server.model import SimulatedLLM, spend, tokenize


def test_tokenize_splits_on_whitespace():
    assert tokenize("a  b\tc\n") == ["a", "b", "c"]


def test_generation_is_deterministic(fast_settings):
    m = SimulatedLLM(fast_settings)
    a = list(m.generate_stream("hello world", 10))
    b = list(m.generate_stream("hello world", 10))
    assert a == b
    assert len(a) == 10


def test_different_prompts_diverge(fast_settings):
    m = SimulatedLLM(fast_settings)
    assert list(m.generate_stream("one", 16)) != list(m.generate_stream("two", 16))


def test_decode_step_skips_finished_sequences(fast_settings):
    m = SimulatedLLM(fast_settings)
    short, long = m.new_sequence("a", 1), m.new_sequence("b", 3)
    m.prefill([short, long])
    assert len(m.decode_step([short, long])) == 2
    assert short.done
    assert len(m.decode_step([short, long])) == 1
    assert len(long.generated) == 2


def test_spend_takes_roughly_requested_time():
    t0 = time.perf_counter()
    spend(20, cpu_fraction=0.5)
    elapsed = (time.perf_counter() - t0) * 1000
    assert 15 <= elapsed < 200
