import asyncio

import pytest

from infer_bench.server.batcher import MicroBatcher, QueueFullError
from infer_bench.server.metrics import ServerMetrics
from infer_bench.server.model import SimulatedLLM

from .conftest import make_settings


async def _collect(it):
    return [t async for t in it]


async def test_batched_output_matches_unbatched():
    settings = make_settings()
    model = SimulatedLLM(settings)
    expected = list(model.generate_stream("same prompt", 12))

    batcher = MicroBatcher(model, settings, ServerMetrics())
    await batcher.start()
    try:
        out = await _collect(await batcher.submit("same prompt", 12))
    finally:
        await batcher.stop()
    assert out == expected


async def test_concurrent_requests_share_batches():
    settings = make_settings(max_batch_size=16, batch_wait_ms=20)
    metrics = ServerMetrics()
    batcher = MicroBatcher(SimulatedLLM(settings), settings, metrics)
    await batcher.start()
    try:
        streams = [await batcher.submit(f"p{i}", 8) for i in range(10)]
        results = await asyncio.gather(*(_collect(s) for s in streams))
    finally:
        await batcher.stop()

    assert all(len(r) == 8 for r in results)
    # 10 requests x 8 tokens decoded in far fewer than 80 steps.
    steps = metrics.registry.get_sample_value("infer_batch_size_count")
    assert steps is not None and steps < 40
    assert metrics.registry.get_sample_value("infer_generated_tokens_total") == 80


async def test_batch_size_is_capped():
    settings = make_settings(max_batch_size=4, batch_wait_ms=20)
    metrics = ServerMetrics()
    batcher = MicroBatcher(SimulatedLLM(settings), settings, metrics)
    await batcher.start()
    try:
        streams = [await batcher.submit(f"p{i}", 4) for i in range(12)]
        await asyncio.gather(*(_collect(s) for s in streams))
    finally:
        await batcher.stop()
    # No observation should land above the 4 bucket.
    assert metrics.registry.get_sample_value("infer_batch_size_bucket", {"le": "4.0"}) == \
        metrics.registry.get_sample_value("infer_batch_size_count")


async def test_queue_full_raises():
    settings = make_settings(max_queue_size=1)
    batcher = MicroBatcher(SimulatedLLM(settings), settings, ServerMetrics())
    # Not started: nothing drains the queue.
    await batcher.submit("a", 1)
    with pytest.raises(QueueFullError):
        await batcher.submit("b", 1)
