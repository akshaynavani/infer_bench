import pytest

from infer_bench.loadgen.backends import get_backend


def test_native_build_and_parse():
    b = get_backend("native")
    call = b.build("hi", 8, stream=True)
    assert call.path == "/v1/generate"
    assert call.body == {"prompt": "hi", "max_new_tokens": 8, "stream": True}
    assert b.parse_response({"completion_tokens": 8}) == 8


def test_sse_parsing_stops_at_done():
    b = get_backend("native")
    lines = [b'data: {"token": "a"}', b"", b": keepalive", b'data: {"token": "b"}',
             b"data: [DONE]", b'data: {"token": "c"}']
    payloads = list(b.iter_sse_payloads(lines))
    assert sum(b.parse_chunk(p) for p in payloads) == 2


def test_openai_completions_chunks():
    b = get_backend("openai", model="m")
    call = b.build("hi", 4, stream=False)
    assert call.path == "/v1/completions" and call.body["model"] == "m"
    assert b.parse_chunk({"choices": [{"text": "tok"}]}) == 1
    assert b.parse_chunk({"choices": [{"text": ""}]}) == 0
    assert b.parse_response({"usage": {"completion_tokens": 4}}) == 4
    assert b.parse_response({"choices": [{"text": "a b c"}]}) == 3


def test_openai_chat_chunks():
    b = get_backend("openai-chat")
    assert b.build("hi", 4, True).body["messages"][0]["content"] == "hi"
    assert b.parse_chunk({"choices": [{"delta": {"content": "x"}}]}) == 1
    assert b.parse_chunk({"choices": [{"delta": {"role": "assistant"}}]}) == 0


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend("grpc")
