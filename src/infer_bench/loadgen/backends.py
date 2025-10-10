"""Adapters that translate a generic generate call into a concrete backend HTTP API.

Each adapter knows how to build the request body for its API and how to pull
generated tokens out of either a JSON response or a server-sent-events stream.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerateCall:
    path: str
    body: dict
    stream: bool


class Backend(ABC):
    name: str = "base"

    def __init__(self, model: str = "simulated"):
        self.model = model

    @abstractmethod
    def build(self, prompt: str, max_tokens: int, stream: bool) -> GenerateCall: ...

    @abstractmethod
    def parse_chunk(self, payload: dict) -> int:
        """Return how many tokens a single decoded SSE payload carries."""

    @abstractmethod
    def parse_response(self, payload: dict) -> int:
        """Return the completion token count from a non-streaming JSON response."""

    def iter_sse_payloads(self, lines):
        """Yield decoded JSON payloads from an iterable of SSE lines (bytes or str)."""
        for raw in lines:
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


class NativeBackend(Backend):
    """The infer_bench reference server's ``/v1/generate`` API."""

    name = "native"

    def build(self, prompt: str, max_tokens: int, stream: bool) -> GenerateCall:
        return GenerateCall(
            path="/v1/generate",
            body={"prompt": prompt, "max_new_tokens": max_tokens, "stream": stream},
            stream=stream,
        )

    def parse_chunk(self, payload: dict) -> int:
        return 1 if "token" in payload else 0

    def parse_response(self, payload: dict) -> int:
        return int(payload.get("completion_tokens", 0))


class OpenAICompletionsBackend(Backend):
    """Any OpenAI-compatible ``/v1/completions`` server (vLLM, TGI, llama.cpp, ...)."""

    name = "openai"

    def build(self, prompt: str, max_tokens: int, stream: bool) -> GenerateCall:
        return GenerateCall(
            path="/v1/completions",
            body={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": 0.0,
                # Force exactly max_tokens so runs are comparable across backends.
                "ignore_eos": True,
            },
            stream=stream,
        )

    def parse_chunk(self, payload: dict) -> int:
        choices = payload.get("choices") or []
        # Most servers emit one token per chunk; count non-empty text pieces.
        return sum(1 for c in choices if c.get("text"))

    def parse_response(self, payload: dict) -> int:
        usage = payload.get("usage") or {}
        if "completion_tokens" in usage:
            return int(usage["completion_tokens"])
        text = "".join(c.get("text", "") for c in payload.get("choices") or [])
        return len(text.split())


class OpenAIChatBackend(OpenAICompletionsBackend):
    """OpenAI-compatible ``/v1/chat/completions``."""

    name = "openai-chat"

    def build(self, prompt: str, max_tokens: int, stream: bool) -> GenerateCall:
        return GenerateCall(
            path="/v1/chat/completions",
            body={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            stream=stream,
        )

    def parse_chunk(self, payload: dict) -> int:
        choices = payload.get("choices") or []
        return sum(1 for c in choices if (c.get("delta") or {}).get("content"))

    def parse_response(self, payload: dict) -> int:
        usage = payload.get("usage") or {}
        if "completion_tokens" in usage:
            return int(usage["completion_tokens"])
        text = "".join(
            (c.get("message") or {}).get("content", "") for c in payload.get("choices") or []
        )
        return len(text.split())


BACKENDS: dict[str, type[Backend]] = {
    b.name: b for b in (NativeBackend, OpenAICompletionsBackend, OpenAIChatBackend)
}


def get_backend(name: str, model: str = "simulated") -> Backend:
    try:
        return BACKENDS[name](model=model)
    except KeyError as exc:
        raise ValueError(f"unknown backend '{name}', choose from {sorted(BACKENDS)}") from exc
