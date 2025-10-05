"""Async continuous micro-batcher.

Incoming requests are put on a queue. A single background task owns the model and loops:

1. admit waiting requests until ``max_batch_size`` sequences are active (waiting up to
   ``batch_wait_ms`` for company when the batch would otherwise be tiny),
2. run prefill for newly admitted sequences,
3. run one decode step over every active sequence, and
4. fan the new tokens back out to each request's token queue.

Model calls run in a worker thread so the event loop keeps accepting connections.
Finished sequences leave the batch immediately and new ones join on the next step
(iteration-level / "continuous" batching).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import ServerSettings
from .metrics import ServerMetrics
from .model import Sequence_, SimulatedLLM

_DONE = object()


class QueueFullError(RuntimeError):
    pass


@dataclass
class _Request:
    seq: Sequence_
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    emitted: int = 0


class MicroBatcher:
    def __init__(self, model: SimulatedLLM, settings: ServerSettings, metrics: ServerMetrics):
        self.model = model
        self.s = settings
        self.metrics = metrics
        self._pending: asyncio.Queue[_Request] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._active: list[_Request] = []
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._loop(), name="micro-batcher")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, prompt: str, max_new_tokens: int) -> AsyncIterator[str]:
        """Enqueue a request and yield its tokens as they are produced."""
        req = _Request(seq=self.model.new_sequence(prompt, max_new_tokens))
        try:
            self._pending.put_nowait(req)
        except asyncio.QueueFull as exc:
            raise QueueFullError("request queue is full") from exc
        self.metrics.queue_depth.set(self._pending.qsize())
        return self._drain(req)

    @staticmethod
    async def _drain(req: _Request) -> AsyncIterator[str]:
        while True:
            item = await req.out.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _admit(self) -> list[_Request]:
        admitted: list[_Request] = []
        capacity = self.s.max_batch_size - len(self._active)
        if capacity <= 0:
            return admitted
        if not self._active and self._pending.empty():
            # Idle: block until at least one request arrives.
            admitted.append(await self._pending.get())
            capacity -= 1
            deadline = time.perf_counter() + self.s.batch_wait_ms / 1000.0
            # Give concurrent arrivals a short window to join the first batch.
            while capacity > 0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    admitted.append(await asyncio.wait_for(self._pending.get(), remaining))
                    capacity -= 1
                except asyncio.TimeoutError:
                    break
        while capacity > 0 and not self._pending.empty():
            admitted.append(self._pending.get_nowait())
            capacity -= 1
        self.metrics.queue_depth.set(self._pending.qsize())
        return admitted

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                new = await self._admit()
                if new:
                    seqs = [r.seq for r in new]
                    await asyncio.to_thread(self.model.prefill, seqs)
                    self.metrics.prompt_tokens.inc(sum(len(s.prompt_tokens) for s in seqs))
                    self._active.extend(new)
                if not self._active:
                    continue

                self.metrics.active_sequences.set(len(self._active))
                self.metrics.batch_size.observe(len(self._active))
                t0 = time.perf_counter()
                await asyncio.to_thread(self.model.decode_step, [r.seq for r in self._active])
                self.metrics.step_latency.observe(time.perf_counter() - t0)

                still_active = []
                for req in self._active:
                    new_tokens = req.seq.generated[req.emitted :]
                    for tok in new_tokens:
                        req.out.put_nowait(tok)
                    req.emitted = len(req.seq.generated)
                    self.metrics.generated_tokens.inc(len(new_tokens))
                    if req.seq.done:
                        req.out.put_nowait(_DONE)
                    else:
                        still_active.append(req)
                self._active = still_active
                self.metrics.active_sequences.set(len(self._active))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                for req in self._active:
                    req.out.put_nowait(exc)
                self._active = []
