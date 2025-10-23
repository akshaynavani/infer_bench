import random

import pytest

from infer_bench.loadgen.workload import LengthDist, Workload, WorkloadSampler


@pytest.mark.parametrize("dist", ["fixed", "uniform", "normal", "lognormal"])
def test_length_dist_respects_bounds(dist):
    d = LengthDist(dist=dist, value=50, min=10, max=100, mean=50, std=40, sigma=1.0)
    rng = random.Random(0)
    xs = [d.sample(rng) for _ in range(500)]
    assert min(xs) >= 10 and max(xs) <= 100


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        LengthDist(dist="zipf").sample(random.Random(0))


def test_from_dict_accepts_int_shorthand():
    w = Workload.from_dict({"prompt_tokens": 32, "max_new_tokens": 16, "seed": 1})
    s = WorkloadSampler(w)
    prompt, n = s.sample()
    assert len(prompt.split()) == 32
    assert n == 16
    assert s.wait() == 0.0


def test_seeded_samplers_are_reproducible_per_user():
    w = Workload.from_dict({"prompt_tokens": {"dist": "uniform", "min": 1, "max": 50}, "seed": 3})
    assert WorkloadSampler(w, 1).sample() == WorkloadSampler(w, 1).sample()
    assert WorkloadSampler(w, 1).sample() != WorkloadSampler(w, 2).sample()


def test_prompts_file_is_used(tmp_path):
    (tmp_path / "prompts.txt").write_text("alpha beta\ngamma\n")
    wf = tmp_path / "w.yaml"
    wf.write_text("prompts_file: prompts.txt\nprompt_tokens: 5\nseed: 1\n")
    w = Workload.load(wf)
    prompt = WorkloadSampler(w).prompt()
    assert len(prompt.split()) == 5
    assert set(prompt.split()) <= {"alpha", "beta", "gamma"}


def test_bundled_workloads_parse():
    from pathlib import Path

    for path in (Path(__file__).parent.parent / "workloads").glob("*.yaml"):
        w = Workload.load(path)
        WorkloadSampler(w).sample()
