"""A deterministic, dependency-free TTS adapter for hermetic testing.

Shipped in the package (not under ``tests/``) on purpose: CI runs the BUILT
``:tts`` image with ``CIS_ADAPTER=stub`` and exercises ``/health``, ``/info``
and ``/v1/audio/speech`` end to end without a GPU and without pulling the
real (CC-BY-NC, ~1 GB) OmniVoice weights. It imports neither torch nor
omnivoice.

Knobs (all via ``CIS_EXTRA`` so the image smoke can set them, plus direct
attributes for in-process tests):

    load_delay_s     sleep inside load()           (default 0)
    synth_delay_s    sleep inside synthesize()     (default 0)
    fail_load        raise inside load()           (default false)
    sample_rate      output rate                   (default 24000)
    duration_s       length of the synthesized tone (default 0.25)

``instructions`` starting with ``invalid:`` raise ``InvalidRequestError`` --
the same 400 path OmniVoice's closed instruct vocabulary produces.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from cis.adapters.base import Synthesis, SynthesisRequest
from cis.config import Config
from cis.errors import InvalidRequestError

VOICES = ["auto", "stub-a", "stub-b"]


class StubAdapter:
    family = "stub"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        x = cfg.extra
        self.load_delay_s = float(x.get("load_delay_s", 0))
        self.synth_delay_s = float(x.get("synth_delay_s", 0))
        self.fail_load = bool(x.get("fail_load", False))
        self.sample_rate = int(x.get("sample_rate", 24000))
        self.duration_s = float(x.get("duration_s", 0.25))

        # Test seams.
        self.load_gate: threading.Event | None = None  # load() blocks until set
        self.load_calls = 0
        self.synth_calls = 0
        self.max_concurrent_synth = 0
        self._active = 0
        self._lock = threading.Lock()
        self._loaded = False

    # -- TTSAdapter ---------------------------------------------------------

    def load(self) -> Any:
        with self._lock:
            self.load_calls += 1
        if self.load_gate is not None:
            self.load_gate.wait()
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
        if self.fail_load:
            raise RuntimeError("stub adapter configured to fail load (fail_load=true)")
        self._loaded = True
        return self

    def voices(self) -> list[str]:
        return list(VOICES)

    def model_version(self) -> str:
        return f"stub-0.0.0+{self.cfg.model}"

    def model_license(self) -> str:
        return "none (synthetic test signal)"

    def describe(self) -> dict[str, Any]:
        return {"device": "cpu", "dtype": "float32", "sample_rate": self.sample_rate}

    def synthesize(self, req: SynthesisRequest) -> Synthesis:
        if not self._loaded:
            raise RuntimeError("stub adapter synthesize() called before load()")
        if req.instructions and req.instructions.startswith("invalid:"):
            raise InvalidRequestError(
                f"Unsupported instruct items found in {req.instructions!r}"
            )
        with self._lock:
            self.synth_calls += 1
            self._active += 1
            self.max_concurrent_synth = max(self.max_concurrent_synth, self._active)
        try:
            if self.synth_delay_s:
                time.sleep(self.synth_delay_s)
            n = int(self.sample_rate * self.duration_s)
            # A 440 Hz tone whose amplitude depends on the voice, so a test can
            # tell voices apart from the bytes alone if it ever needs to.
            amp = 0.5 if req.voice == "auto" else 0.25
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            samples = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            return Synthesis(samples=samples, sample_rate=self.sample_rate)
        finally:
            with self._lock:
                self._active -= 1
