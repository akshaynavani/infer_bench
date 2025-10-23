import json

import pytest
from fastapi.testclient import TestClient

from infer_bench.server.app import create_app
from infer_bench.server.config import ServingMode
from infer_bench.server.metrics import ServerMetrics

from .conftest import make_settings


@pytest.fixture(params=[ServingMode.NAIVE, ServingMode.BATCHED], ids=lambda m: m.value)
def client(request):
    app = create_app(make_settings(mode=request.param), ServerMetrics())
    with TestClient(app) as c:
        yield c


def _sse_payloads(resp):
    out = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            out.append(json.loads(data))
    return out


def test_health_reports_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] in {"naive", "batched"}


def test_generate_json(client):
    r = client.post("/v1/generate", json={"prompt": "hello there", "max_new_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["completion_tokens"] == 5
    assert body["prompt_tokens"] == 2
    assert len(body["text"].split()) == 5


def test_generate_stream(client):
    with client.stream(
        "POST", "/v1/generate", json={"prompt": "hi", "max_new_tokens": 6, "stream": True}
    ) as r:
        assert r.status_code == 200
        payloads = _sse_payloads(r)
    assert len(payloads) == 6
    assert all("token" in p for p in payloads)


def test_openai_completions(client):
    r = client.post("/v1/completions", json={"prompt": "a b c", "max_tokens": 4})
    body = r.json()
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_openai_completions_stream(client):
    with client.stream(
        "POST", "/v1/completions", json={"prompt": "x", "max_tokens": 3, "stream": True}
    ) as r:
        payloads = _sse_payloads(r)
    assert len(payloads) == 3
    assert payloads[0]["object"] == "text_completion"


def test_rejects_too_many_tokens(client):
    r = client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 10_000})
    assert r.status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 3})
    text = client.get("/metrics").text
    assert "infer_requests_total" in text
    assert "infer_generated_tokens_total 3.0" in text
