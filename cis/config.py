"""Env-driven configuration: the ``CIS_*`` contract.

Compose-native environment variables, never a mounted config file -- every
existing AceTeam sidecar is configured by env, and citadel-cli's host-port
injection is env-shaped. ``PORT``, ``HF_HOME`` and ``HF_TOKEN`` are kept
UNPREFIXED for parity with those sidecars (design doc §3.3 / §8).

The pinned cross-phase contract (citadel-cli P1 and aceteam P2 read these):

    CIS_TASK=tts  CIS_ADAPTER=omnivoice  CIS_MODEL=k2-fsa/OmniVoice
    CIS_DEVICE    CIS_DTYPE    CIS_SLOTS    CIS_MAX_INPUT_CHARS
    CIS_PRELOAD   CIS_EXTRA (JSON object)

``Config.from_env`` is the single authority for defaults and parsing; read it
rather than trusting a table elsewhere.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cis.errors import ConfigError

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}

VALID_TASKS = ("tts",)
VALID_DTYPES = ("auto", "float16", "bfloat16", "float32")

DEFAULT_PORT = 8000
DEFAULT_HF_HOME = "/root/.cache/huggingface"
DEFAULT_SLOTS = 2
DEFAULT_MAX_INPUT_CHARS = 5000


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in TRUTHY:
        return True
    if v in FALSY:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/true/yes/on or 0/false/no/off)")


def _parse_int(name: str, raw: str | None, default: int, *, minimum: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not an integer") from e
    if v < minimum:
        raise ConfigError(f"{name}={v} must be >= {minimum}")
    return v


def _parse_extra(raw: str | None) -> dict[str, Any]:
    """``CIS_EXTRA`` is a JSON OBJECT of adapter-specific knobs. Malformed JSON
    refuses startup loudly -- a silently-ignored ``{"num_step": 16`` typo would
    otherwise run every request at the default step count with no signal."""
    if raw is None or raw.strip() == "":
        return {}
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"CIS_EXTRA is not valid JSON: {e.msg} (at char {e.pos})") from e
    if not isinstance(v, dict):
        raise ConfigError(f"CIS_EXTRA must be a JSON object, got {type(v).__name__}")
    return v


@dataclass(frozen=True)
class Config:
    task: str
    adapter: str
    model: str
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    port: int = DEFAULT_PORT
    hf_home: str = DEFAULT_HF_HOME
    hf_token: str | None = None
    slots: int = DEFAULT_SLOTS
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    preload: bool = True
    disk_preflight: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env

        def req(name: str) -> str:
            v = (env.get(name) or "").strip()
            if not v:
                raise ConfigError(f"{name} is required")
            return v

        task = req("CIS_TASK").lower()
        if task not in VALID_TASKS:
            raise ConfigError(
                f"CIS_TASK={task!r} is not supported; valid: {', '.join(VALID_TASKS)}"
            )

        dtype = (env.get("CIS_DTYPE") or "auto").strip().lower()
        if dtype not in VALID_DTYPES:
            raise ConfigError(
                f"CIS_DTYPE={dtype!r} is not supported; valid: {', '.join(VALID_DTYPES)}"
            )

        device = (env.get("CIS_DEVICE") or "auto").strip().lower()
        if not (device in ("auto", "cpu") or device.startswith("cuda")):
            raise ConfigError(
                f"CIS_DEVICE={device!r} is not supported; use auto, cpu, cuda or cuda:N"
            )

        return cls(
            task=task,
            adapter=req("CIS_ADAPTER").lower(),
            model=req("CIS_MODEL"),
            revision=(env.get("CIS_MODEL_REVISION") or "").strip() or None,
            device=device,
            dtype=dtype,
            port=_parse_int("PORT", env.get("PORT"), DEFAULT_PORT, minimum=1),
            hf_home=(env.get("HF_HOME") or "").strip() or DEFAULT_HF_HOME,
            hf_token=(env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
            or None,
            slots=_parse_int("CIS_SLOTS", env.get("CIS_SLOTS"), DEFAULT_SLOTS, minimum=1),
            max_input_chars=_parse_int(
                "CIS_MAX_INPUT_CHARS",
                env.get("CIS_MAX_INPUT_CHARS"),
                DEFAULT_MAX_INPUT_CHARS,
                minimum=1,
            ),
            preload=_parse_bool("CIS_PRELOAD", env.get("CIS_PRELOAD"), True),
            disk_preflight=_parse_bool("CIS_DISK_PREFLIGHT", env.get("CIS_DISK_PREFLIGHT"), True),
            extra=_parse_extra(env.get("CIS_EXTRA")),
        )
