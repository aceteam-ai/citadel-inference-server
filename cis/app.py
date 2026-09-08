"""FastAPI app factory: the runtime core's HTTP surface.

    GET /health  -> {"status": "up"|"loading", "model_loaded": bool, ...}
                    ALWAYS HTTP 200. citadel-cli's synthesizeHealthReady gates
                    on the BODY's model_loaded, never the status code; a 503
                    during load would trip its connection-refused budget.
    GET /info    -> {"model", "adapter", "task", "model_license", "voices", ...}

Task routes (``/v1/audio/speech`` for ``tts``) are mounted from the task
module selected by ``CIS_TASK``; the adapter family comes from
``CIS_ADAPTER`` via ``cis.adapters.create``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from cis import __version__, adapters
from cis.adapters import tts as tts_task
from cis.config import Config
from cis.loader import ModelLoader, exit_process

logger = logging.getLogger("cis.app")


def create_app(
    cfg: Config | None = None,
    *,
    adapter: Any | None = None,
    on_fatal: Callable[[BaseException], None] = exit_process,
) -> FastAPI:
    cfg = cfg or Config.from_env()
    adapter = adapter if adapter is not None else adapters.create(cfg)
    loader = ModelLoader(adapter.load, on_fatal=on_fatal)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "cis %s: task=%s adapter=%s model=%s device=%s dtype=%s slots=%d preload=%s",
            __version__, cfg.task, cfg.adapter, cfg.model, cfg.device, cfg.dtype,
            cfg.slots, cfg.preload,
        )
        if cfg.preload:
            # Background, AFTER the socket is bound: /health serves "loading"
            # while weights download + load (see cis.loader for why).
            loader.start_preload()
        yield

    app = FastAPI(title="citadel-inference-server", version=__version__, lifespan=lifespan)
    app.state.cfg = cfg
    app.state.adapter = adapter
    app.state.loader = loader

    @app.get("/health")
    async def health() -> dict[str, Any]:
        loaded = loader.loaded
        body: dict[str, Any] = {
            "status": "up" if loaded else "loading",
            "model_loaded": loaded,
            "model_version": adapter.model_version(),
            "slots": cfg.slots,
            "device": adapter.describe().get("device"),
        }
        if loader.error is not None:
            body["last_error"] = f"{type(loader.error).__name__}: {loader.error}"
        return body

    @app.get("/info")
    async def info() -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": cfg.model,
            "adapter": cfg.adapter,
            "task": cfg.task,
            "model_license": adapter.model_license(),
            # task-level fields (voices, formats, capacity) -- voices is pinned
            "model_loaded": loader.loaded,
            "model_version": adapter.model_version(),
            "revision": cfg.revision,
            "cis_version": __version__,
            "load_seconds": loader.load_seconds,
        }
        body.update(tts_task.info_fields(cfg, adapter))
        body["adapter_info"] = adapter.describe()
        return body

    if cfg.task == "tts":
        app.include_router(tts_task.build_router(cfg, adapter, loader))
    else:  # pragma: no cover -- Config.from_env already rejects unknown tasks
        raise ValueError(f"no task module for CIS_TASK={cfg.task!r}")

    return app
