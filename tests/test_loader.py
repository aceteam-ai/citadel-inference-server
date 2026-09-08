"""ModelLoader in isolation: single-flight, retry after failure, fatal preload."""

import asyncio
import threading

import pytest

from cis.loader import LoadState, ModelLoader


def run(coro):
    return asyncio.run(coro)


def test_single_flight_concurrent_get():
    calls = 0
    gate = threading.Event()

    def load():
        nonlocal calls
        calls += 1
        gate.wait()
        return "model"

    async def main():
        loader = ModelLoader(load)
        tasks = [asyncio.create_task(loader.get()) for _ in range(5)]
        await asyncio.sleep(0.05)
        assert loader.state is LoadState.LOADING
        gate.set()
        results = await asyncio.gather(*tasks)
        assert results == ["model"] * 5
        assert loader.loaded
        assert await loader.get() == "model"

    run(main())
    assert calls == 1


def test_failure_propagates_to_all_waiters_and_allows_retry():
    attempts = 0

    def load():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return "ok"

    async def main():
        loader = ModelLoader(load)
        results = await asyncio.gather(
            loader.get(), loader.get(), loader.get(), return_exceptions=True
        )
        assert all(isinstance(r, RuntimeError) for r in results)
        assert loader.state is LoadState.FAILED
        assert "boom" in str(loader.error)
        assert await loader.get() == "ok"
        assert loader.loaded and loader.error is None

    run(main())
    assert attempts == 2


def test_preload_failure_calls_on_fatal():
    seen = []

    def load():
        raise RuntimeError("no weights")

    async def main():
        loader = ModelLoader(load, on_fatal=seen.append)
        task = loader.start_preload()
        with pytest.raises(RuntimeError):
            await task
        await asyncio.sleep(0)  # let the done-callback run

    run(main())
    assert len(seen) == 1 and "no weights" in str(seen[0])


def test_preload_success_no_fatal_and_get_joins():
    seen = []

    async def main():
        loader = ModelLoader(lambda: "m", on_fatal=seen.append)
        loader.start_preload()
        loader.start_preload()  # idempotent
        assert await loader.get() == "m"
        assert loader.load_seconds is not None

    run(main())
    assert seen == []


def test_get_shielded_from_caller_cancellation():
    gate = threading.Event()
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        gate.wait()
        return "m"

    async def main():
        loader = ModelLoader(load)
        t1 = asyncio.create_task(loader.get())
        await asyncio.sleep(0.02)
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1
        assert loader.state is LoadState.LOADING, "the shared load survives one caller's cancel"
        gate.set()
        assert await loader.get() == "m"

    run(main())
    assert calls == 1
