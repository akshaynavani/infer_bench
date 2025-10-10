"""FastAPI reference inference server.

Run with ``infer-bench serve`` or ``uvicorn infer_bench.server.app:app``.
The serving mode (naive vs batched) is selected via ``INFER_BENCH_MODE``.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from .batcher import MicroBatcher, QueueFullError
from .config import ServerSettings, ServingMode
from .metrics import ServerMetrics
from .model import build_backend, tokenize


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1)
    stream: bool = False


class GenerateResponse(BaseModel):
    id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    latency_ms: float


class CompletionRequest(BaseModel):
    """Subset of the OpenAI /v1/completions schema."""

    model: str = "simulated"
    prompt: str
    max_tokens: int = Field(default=64, ge=1)
    stream: bool = False


def create_app(settings: ServerSettings | None = None, metrics: ServerMetrics | None = None):
    settings = settings or ServerSettings()
    metrics = metrics or ServerMetrics()
    model = build_backend(settings)
    batcher = MicroBatcher(model, settings, metrics) if settings.mode == ServingMode.BATCHED else None
    mode = settings.mode.value

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if batcher is not None:
            await batcher.start()
        yield
        if batcher is not None:
            await batcher.stop()

    app = FastAPI(title="infer_bench reference server", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.metrics = metrics

    async def token_source(prompt: str, max_new_tokens: int) -> AsyncIterator[str]:
        if max_new_tokens > settings.max_new_tokens_limit:
            raise HTTPException(422, f"max_new_tokens exceeds {settings.max_new_tokens_limit}")
        if batcher is not None:
            try:
                return await batcher.submit(prompt, max_new_tokens)
            except QueueFullError as exc:
                metrics.requests.labels(mode, "rejected").inc()
                raise HTTPException(503, str(exc)) from exc
        # Naive mode: run the whole single-sequence generation loop on the threadpool.
        return iterate_in_threadpool(model.generate_stream(prompt, max_new_tokens))

    async def instrumented(prompt: str, tokens: AsyncIterator[str]) -> AsyncIterator[tuple[str, float]]:
        """Yield (token, seconds_since_start) while recording server metrics."""
        start = time.perf_counter()
        first = True
        n = 0
        metrics.in_flight.inc()
        status = "ok"
        try:
            async for tok in tokens:
                now = time.perf_counter() - start
                if first:
                    metrics.ttft.labels(mode).observe(now)
                    first = False
                n += 1
                yield tok, now
        except BaseException:
            status = "error"
            raise
        finally:
            metrics.in_flight.dec()
            metrics.requests.labels(mode, status).inc()
            metrics.latency.labels(mode).observe(time.perf_counter() - start)
            if batcher is None:
                # The batcher records token counters itself.
                metrics.generated_tokens.inc(n)
                metrics.prompt_tokens.inc(len(tokenize(prompt)))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "mode": mode, "backend": settings.backend.value}

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/generate", response_model=None)
    async def generate(req: GenerateRequest):
        req_id = uuid.uuid4().hex
        tokens = instrumented(req.prompt, await token_source(req.prompt, req.max_new_tokens))

        if req.stream:

            async def sse() -> AsyncIterator[bytes]:
                async for tok, _ in tokens:
                    yield f"data: {json.dumps({'id': req_id, 'token': tok})}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        start = time.perf_counter()
        out: list[str] = []
        ttft = 0.0
        async for tok, t in tokens:
            if not out:
                ttft = t
            out.append(tok)
        return GenerateResponse(
            id=req_id,
            text=" ".join(out),
            prompt_tokens=len(tokenize(req.prompt)),
            completion_tokens=len(out),
            ttft_ms=ttft * 1000,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest):
        """Minimal OpenAI-compatible completions endpoint."""
        req_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        tokens = instrumented(req.prompt, await token_source(req.prompt, req.max_tokens))

        if req.stream:

            async def sse() -> AsyncIterator[bytes]:
                async for tok, _ in tokens:
                    chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "text": tok + " ", "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        out = [tok async for tok, _ in tokens]
        prompt_tokens = len(tokenize(req.prompt))
        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "text": " ".join(out), "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(out),
                "total_tokens": prompt_tokens + len(out),
            },
        }

    return app


def __getattr__(name: str):
    # Lazily build the module-level ``app`` so importing this module (e.g. in tests)
    # does not construct a server with environment-derived settings.
    if name == "app":
        return create_app()
    raise AttributeError(name)
