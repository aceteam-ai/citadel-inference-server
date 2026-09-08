"""Adapter registry: ``(task, family)`` -> constructor.

``create(cfg)`` picks the family from ``CIS_ADAPTER`` and refuses when its task
does not match ``CIS_TASK`` -- a ``CIS_TASK=tts`` with a future image adapter
must fail at startup, not mount the wrong routes. Family modules are imported
lazily so an unused family's (heavy) dependencies are never touched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cis.config import Config
from cis.errors import ConfigError

# family -> (task, factory). Factories take the Config.
_FAMILIES: dict[str, tuple[str, Callable[[Config], Any]]] = {}


def register(family: str, task: str, factory: Callable[[Config], Any]) -> None:
    _FAMILIES[family] = (task, factory)


def _omnivoice(cfg: Config) -> Any:
    from cis.adapters.omnivoice import OmniVoiceAdapter

    return OmniVoiceAdapter(cfg)


def _stub(cfg: Config) -> Any:
    from cis.adapters.stub import StubAdapter

    return StubAdapter(cfg)


register("omnivoice", "tts", _omnivoice)
register("stub", "tts", _stub)


def families() -> dict[str, str]:
    """family -> task, for error messages and ``/info``."""
    return {k: v[0] for k, v in _FAMILIES.items()}


def create(cfg: Config) -> Any:
    entry = _FAMILIES.get(cfg.adapter)
    if entry is None:
        raise ConfigError(
            f"CIS_ADAPTER={cfg.adapter!r} is not a known adapter family; "
            f"known: {', '.join(sorted(_FAMILIES))}"
        )
    task, factory = entry
    if task != cfg.task:
        raise ConfigError(
            f"CIS_ADAPTER={cfg.adapter!r} implements task {task!r}, but CIS_TASK={cfg.task!r}"
        )
    try:
        return factory(cfg)
    except ValueError as e:
        raise ConfigError(f"adapter {cfg.adapter!r} rejected its configuration: {e}") from e
