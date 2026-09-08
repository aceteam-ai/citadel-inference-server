"""Hermetic fixtures: every test drives the app through the ``stub`` adapter
(or a fake OmniVoice class). No GPU, no network, no model download."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cis.adapters.stub import StubAdapter
from cis.app import create_app
from cis.config import Config

BASE_ENV: dict[str, str] = {
    "CIS_TASK": "tts",
    "CIS_ADAPTER": "stub",
    "CIS_MODEL": "stub/test-model",
    "CIS_DEVICE": "cpu",
}


def make_config(**overrides: Any) -> Config:
    """Build a Config from BASE_ENV plus CIS_* overrides (given as env strings
    or, for ``extra``, a dict that is JSON-encoded into CIS_EXTRA)."""
    env = dict(BASE_ENV)
    for k, v in overrides.items():
        if k == "extra":
            env["CIS_EXTRA"] = json.dumps(v)
        else:
            env[k] = str(v)
    return Config.from_env(env)


def fatal_recorder() -> tuple[list[BaseException], Any]:
    seen: list[BaseException] = []

    def on_fatal(exc: BaseException) -> None:
        seen.append(exc)

    return seen, on_fatal


@pytest.fixture
def stub_app_factory():
    """Returns ``build(**cfg_overrides) -> (client, adapter, fatal_list)``.
    The TestClient is entered (lifespan runs) and closed on teardown."""
    clients: list[TestClient] = []

    def build(**overrides: Any):
        cfg = make_config(**overrides)
        adapter = StubAdapter(cfg)
        fatal, on_fatal = fatal_recorder()
        app = create_app(cfg, adapter=adapter, on_fatal=on_fatal)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, adapter, fatal

    yield build
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(stub_app_factory) -> Iterator[TestClient]:
    c, _adapter, _fatal = stub_app_factory()
    yield c
