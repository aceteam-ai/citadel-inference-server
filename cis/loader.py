"""Lazy, single-flight model loading.

One load, ever, per process: N concurrent first requests all await the SAME
in-flight load instead of triggering N loads (which on a GPU means N copies of
the weights racing for VRAM). The blocking ``load_fn`` runs in a worker thread
(``asyncio.to_thread``) so the event loop -- and therefore ``/health`` -- keeps
serving while weights are pulled and loaded.

Two entry points:

* ``get()`` -- the lazy path. Starts the load on first call, joins it on every
  subsequent call while it is in flight. A FAILED load resets the state so the
  next ``get()`` retries (a transient download error should not brick the
  process for its whole lifetime); the caller sees the exception.
* ``start_preload()`` -- the ``CIS_PRELOAD=true`` path, called from the app's
  lifespan AFTER uvicorn has bound its socket. This is deliberately NOT the
  kokoro shape (load synchronously inside lifespan): citadel-cli's
  ``SynthesizeSpeechHandler`` gives a connection-refused server only ~8s but a
  reachable-but-``model_loaded:false`` server ~120s, so binding first is what
  makes ``{"status":"loading"}`` mean anything on a first-start pull. A preload
  FAILURE is fatal by default (``on_fatal`` exits the process non-zero so the
  compose ``restart:`` policy retries cleanly) -- a server that reports
  "loading" forever is the green-but-wedged shape this codebase's node-side
  watchdogs exist to catch, and there is no operator to notice a log line.

All state transitions happen on the event loop between awaits, so no lock is
needed; the only cross-thread boundary is the ``to_thread`` call itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger("cis.loader")


class LoadState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


def exit_process(exc: BaseException) -> None:
    """Default ``on_fatal``: log and hard-exit. ``os._exit`` rather than
    ``sys.exit`` because this runs inside an asyncio task callback, where
    ``SystemExit`` would just be swallowed by the loop's exception handler."""
    logger.critical("fatal: model preload failed: %s: %s", type(exc).__name__, exc)
    sys.stderr.flush()
    os._exit(1)


class ModelLoader:
    def __init__(
        self,
        load_fn: Callable[[], Any],
        *,
        on_fatal: Callable[[BaseException], None] = exit_process,
    ) -> None:
        self._load_fn = load_fn
        self._on_fatal = on_fatal
        self._task: asyncio.Task[Any] | None = None
        self._state = LoadState.IDLE
        self._error: BaseException | None = None
        self._load_seconds: float | None = None

    @property
    def state(self) -> LoadState:
        return self._state

    @property
    def loaded(self) -> bool:
        return self._state is LoadState.LOADED

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    async def _run(self) -> Any:
        self._state = LoadState.LOADING
        self._error = None
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(self._load_fn)
        except BaseException as e:  # noqa: BLE001 -- recorded, then re-raised to awaiters
            self._state = LoadState.FAILED
            self._error = e
            self._task = None  # allow a later get() to retry
            logger.error("model load failed after %.1fs: %s: %s",
                         time.monotonic() - started, type(e).__name__, e)
            raise
        self._load_seconds = round(time.monotonic() - started, 3)
        self._state = LoadState.LOADED
        logger.info("model loaded in %.1fs", self._load_seconds)
        return result

    def _ensure_task(self) -> asyncio.Task[Any]:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="cis-model-load")
        return self._task

    async def get(self) -> Any:
        """Return the loaded model, loading it (once) if needed. Raises the
        load's exception to every concurrent awaiter if it fails."""
        if self._state is LoadState.LOADED and self._task is not None:
            return self._task.result()
        task = self._ensure_task()
        # shield: a client disconnect cancelling THIS request must not cancel
        # the shared load every other waiter is joined on.
        return await asyncio.shield(task)

    def start_preload(self) -> asyncio.Task[Any]:
        """Kick off the load in the background (idempotent). A failure invokes
        ``on_fatal`` -- by default, process exit."""
        task = self._ensure_task()

        def _done(t: asyncio.Task[Any]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self._on_fatal(exc)

        task.add_done_callback(_done)
        return task
