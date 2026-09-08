"""OmniVoice (``k2-fsa/OmniVoice``) TTS adapter -- the first consumer.

Verified against upstream omnivoice 0.2.1 (``omnivoice/models/omnivoice.py``):

* ``OmniVoice.from_pretrained(path, device_map=<device>, dtype=<torch dtype>)``
  -- ``dtype=``, not ``torch_dtype=`` (transformers >= 5). ``load_asr``
  defaults False; the Whisper ASR model is only ever constructed on the
  voice-CLONING path (``ref_audio`` without ``ref_text``), which this adapter
  never takes.
* ``model.generate(text=, language=, instruct=, speed=, num_step=, ...)``
  returns a LIST of 1-D ``np.ndarray`` at ``model.sampling_rate`` (24 kHz).
* ``instruct`` is NOT free text. ``_resolve_instruct`` validates each
  comma-separated item against a closed vocabulary (gender / age / pitch /
  whisper / English accent / Chinese dialect) and raises ``ValueError`` for
  anything else -- with a message that lists the valid items. This adapter
  passes ``instructions`` through verbatim and maps that ``ValueError`` to a
  400 rather than re-implementing (and drifting from) the upstream validator.
  Consequently every ``voice`` PRESET below is composed only of vocabulary
  items upstream actually accepts.

Contract mapping (design doc §6.3, with the brief's decisions applied):

    input                    -> text=
    voice: "auto" (default)  -> no instruct: the model's own default speaker
    voice: <preset>          -> instruct=PRESETS[preset]
    instructions             -> instruct= verbatim; OVERRIDES the preset
    language                 -> language=
    speed (0.5..2.0)         -> speed= (only when the caller sent one)
    CIS_EXTRA.num_step       -> num_step= (default 32)
    output                   -> audios[0] @ model.sampling_rate

**Voice cloning is deferred to v2. This adapter never passes ``ref_audio``,
``ref_text`` or ``voice_clone_prompt`` to ``generate`` and never passes
``load_asr``/``asr_*`` to ``from_pretrained``** -- pinned by
``tests/test_omnivoice_adapter.py``, which asserts key ABSENCE.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from cis import hub
from cis.adapters.base import Synthesis, SynthesisRequest
from cis.config import Config
from cis.errors import InvalidRequestError

logger = logging.getLogger("cis.adapters.omnivoice")

DEFAULT_NUM_STEP = 32

# Closed preset vocabulary -> upstream instruct string. Every value here is a
# comma+space-separated list of items from omnivoice's _INSTRUCT_CATEGORIES
# (see the module docstring); anything else would 400 at generate time.
PRESETS: dict[str, str] = {
    "female": "female",
    "male": "male",
    "female-young": "female, young adult",
    "male-young": "male, young adult",
    "female-mature": "female, middle-aged",
    "male-mature": "male, middle-aged",
    "female-high-pitch": "female, high pitch",
    "male-low-pitch": "male, low pitch",
    "female-american": "female, american accent",
    "male-american": "male, american accent",
    "female-british": "female, british accent",
    "male-british": "male, british accent",
    "whisper": "whisper",
}

VOICES: list[str] = ["auto", *PRESETS]

# The checkpoint's license exactly as the model card states it. The card says
# "CC-BY-NC" with NO version and carries no SPDX `license:` tag in its
# metadata, so no version suffix is invented here. Any other CIS_MODEL is
# reported as "unknown" unless CIS_EXTRA.model_license overrides it.
KNOWN_MODEL_LICENSES: dict[str, str] = {
    "k2-fsa/OmniVoice": "CC-BY-NC",
}


def _import_omnivoice_class() -> Any:
    from omnivoice import OmniVoice  # heavy: torch + transformers

    return OmniVoice


def _import_torch() -> Any:
    import torch

    return torch


def omnivoice_package_version() -> str:
    try:
        return version("omnivoice")
    except PackageNotFoundError:
        return "unknown"


def resolve_device(torch_mod: Any, requested: str) -> str:
    """``auto`` -> CUDA when visible, else CPU. An explicit ``cuda`` on a box
    with no visible GPU is a hard error (kokoro's rule): silently serving a
    32-step diffusion LM on CPU would "start fine and time out every request",
    which is worse than failing loudly at load."""
    if requested == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch_mod.cuda.is_available():
        if getattr(torch_mod.version, "cuda", None) is None:
            raise RuntimeError(
                f"CIS_DEVICE={requested} but this torch build has no CUDA support "
                "(torch.version.cuda is None). Use the CUDA image, or set CIS_DEVICE=cpu."
            )
        raise RuntimeError(
            f"CIS_DEVICE={requested} and torch is a CUDA build "
            f"(torch.version.cuda={torch_mod.version.cuda}), but no GPU is visible to "
            "this container. Expose one with the nvidia runtime (compose "
            "deploy.resources.reservations.devices) and confirm the host driver + "
            "container toolkit are installed, or set CIS_DEVICE=cpu."
        )
    return requested


def resolve_dtype(torch_mod: Any, requested: str, device: str) -> Any:
    """``auto`` = fp16 on CUDA, fp32 on CPU (the kokoro/diffusers rule)."""
    if requested == "auto":
        requested = "float16" if device.startswith("cuda") else "float32"
    return getattr(torch_mod, requested)


class OmniVoiceAdapter:
    family = "omnivoice"

    def __init__(
        self,
        cfg: Config,
        *,
        omnivoice_cls_fn: Callable[[], Any] = _import_omnivoice_class,
        torch_fn: Callable[[], Any] = _import_torch,
        ensure_snapshot_fn: Callable[..., str] = hub.ensure_snapshot,
    ) -> None:
        self.cfg = cfg
        self._omnivoice_cls_fn = omnivoice_cls_fn
        self._torch_fn = torch_fn
        self._ensure_snapshot_fn = ensure_snapshot_fn
        self._model: Any = None
        self._device: str = cfg.device
        self._dtype_name: str = cfg.dtype
        try:
            self._num_step = int(cfg.extra.get("num_step", DEFAULT_NUM_STEP))
        except (TypeError, ValueError) as e:
            raise ValueError(f"CIS_EXTRA.num_step must be an integer: {e}") from e
        if self._num_step < 1:
            raise ValueError("CIS_EXTRA.num_step must be >= 1")

    # -- TTSAdapter ---------------------------------------------------------

    def load(self) -> Any:
        torch = self._torch_fn()
        omnivoice_cls = self._omnivoice_cls_fn()

        device = resolve_device(torch, self.cfg.device)
        dtype = resolve_dtype(torch, self.cfg.dtype, device)
        self._device = device
        self._dtype_name = str(dtype).removeprefix("torch.")

        local_path = self._ensure_snapshot_fn(
            self.cfg.model,
            revision=self.cfg.revision,
            token=self.cfg.hf_token,
            disk_preflight=self.cfg.disk_preflight,
        )
        logger.info("loading OmniVoice from %s on %s (%s), num_step=%d",
                    local_path, device, self._dtype_name, self._num_step)
        # NEVER pass load_asr / asr_model_name / asr_device here (v2 cloning).
        self._model = omnivoice_cls.from_pretrained(local_path, device_map=device, dtype=dtype)
        return self._model

    def voices(self) -> list[str]:
        return list(VOICES)

    def model_version(self) -> str:
        return f"omnivoice-{omnivoice_package_version()}+{self.cfg.model}"

    def model_license(self) -> str:
        override = self.cfg.extra.get("model_license")
        if isinstance(override, str) and override.strip():
            return override.strip()
        return KNOWN_MODEL_LICENSES.get(self.cfg.model, "unknown")

    def describe(self) -> dict[str, Any]:
        return {
            "device": self._device,
            "dtype": self._dtype_name,
            "num_step": self._num_step,
            "sample_rate": int(self._model.sampling_rate) if self._model is not None else None,
            "voice_cloning": False,  # v2
        }

    def generate_kwargs(self, req: SynthesisRequest) -> dict[str, Any]:
        """The exact kwargs handed to ``model.generate`` -- factored out so the
        mapping is testable without a model."""
        kwargs: dict[str, Any] = {"text": req.text, "num_step": self._num_step}
        instruct = (req.instructions or "").strip() or PRESETS.get(req.voice)
        if instruct:
            kwargs["instruct"] = instruct
        if req.language:
            kwargs["language"] = req.language
        if req.speed is not None:
            kwargs["speed"] = float(req.speed)
        return kwargs

    def synthesize(self, req: SynthesisRequest) -> Synthesis:
        if self._model is None:
            raise RuntimeError("OmniVoice adapter synthesize() called before load()")
        kwargs = self.generate_kwargs(req)
        try:
            audios = self._model.generate(**kwargs)
        except ValueError as e:
            # omnivoice's closed instruct vocabulary (or a bad language id).
            raise InvalidRequestError(str(e)) from e
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        samples = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        return Synthesis(samples=samples, sample_rate=int(self._model.sampling_rate))
