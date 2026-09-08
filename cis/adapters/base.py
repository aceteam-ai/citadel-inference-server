"""Adapter protocols. A TASK decides which HTTP routes mount (``tts`` ->
``/v1/audio/speech``); an adapter FAMILY decides how the model is called.
Adapters never touch HTTP, config parsing, caching or health -- those are the
runtime core's job (design doc §3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SynthesisRequest:
    """The task layer's already-validated view of a ``/v1/audio/speech`` body.
    ``voice`` is guaranteed to be one of ``TTSAdapter.voices()``; the format
    has already been checked; ``text`` is within ``CIS_MAX_INPUT_CHARS``."""

    text: str
    voice: str = "auto"
    instructions: str | None = None
    language: str | None = None
    speed: float | None = None


@dataclass(frozen=True)
class Synthesis:
    samples: np.ndarray  # 1-D float32 PCM in [-1, 1]
    sample_rate: int


@runtime_checkable
class TTSAdapter(Protocol):
    family: str

    def load(self) -> Any:
        """Weights -> device. Blocking; the loader runs it in a thread."""

    def voices(self) -> list[str]:
        """The closed, enumerable voice vocabulary advertised on ``/info``.
        ``"auto"`` must be first."""

    def model_version(self) -> str:
        """``<family>-<pkg-version>+<model>`` -- the ``X-TTS-Model-Version``
        receipt component (kokoro's shape: ``kokoro-0.9.4+hexgrad/Kokoro-82M``)."""

    def model_license(self) -> str:
        """The served CHECKPOINT's license as the model card states it."""

    def describe(self) -> dict[str, Any]:
        """Additive, family-specific fields merged into ``/info`` (resolved
        device/dtype, generation knobs, ...)."""

    def synthesize(self, req: SynthesisRequest) -> Synthesis:
        """Blocking. Raise ``InvalidRequestError`` for a family-level request
        rejection (mapped to 400); anything else is a 500."""
