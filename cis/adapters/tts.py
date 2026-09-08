"""The ``tts`` TASK: ``POST /v1/audio/speech``, kokoro-byte-identical.

Request (OpenAI-shaped; kokoro's ``SpeechRequest`` + ``instructions``):

    {"input": <text>, "voice": "auto", "response_format": "wav",
     "instructions": <optional>, "language": <optional>, "speed": <optional>}

* ``voice`` is an ENUMERABLE vocabulary -- ``"auto"`` plus the adapter's
  presets, advertised on ``/info``.``voices``. Unknown -> 400. It never
  carries free text (Jason's decision: enumerable voice + separate
  ``instructions``, matching OpenAI's shape).
* ``instructions`` is the free-form "voice design" field, handed to the
  adapter verbatim; the adapter may reject it (400 with its own message).
* ``response_format`` is ``wav`` only in v1 (straight PCM-16 RIFF from the
  adapter's float samples). opus/mp3 would need ffmpeg and are deliberately
  NOT offered; anything else -> 400.
* ``speed``/``language``/``model`` are optional additive fields so a
  kokoro-shaped caller (which always sends ``speed``) never 422s; ``model`` is
  accepted and ignored exactly as kokoro does.

Response: 200 with the raw audio bytes, ``Content-Type: audio/wav``, and the
five receipt headers citadel-cli's ``synthesizeReceiptFromHeaders`` reads:
``X-TTS-Model-Version``, ``X-TTS-Cache-Key``, ``X-TTS-Chars``,
``X-TTS-Duration-Seconds``, ``X-TTS-Cache-Hit``. There is no content-addressed
audio cache in P0, so ``X-TTS-Cache-Hit`` is always ``"0"``; the key is still
computed over every generation input so a later cache (or a receipt consumer)
gets a stable identifier.

Errors are FastAPI ``HTTPException``s -> ``{"detail": "..."}`` bodies, which is
what kokoro's server actually emits (the design doc's ``{"error": ...}`` shape
was not what kokoro does; the Go handler surfaces the body verbatim either
way).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from cis.adapters.base import Synthesis, SynthesisRequest, TTSAdapter
from cis.config import Config
from cis.errors import InvalidRequestError
from cis.loader import ModelLoader

logger = logging.getLogger("cis.tts")

DEFAULT_VOICE = "auto"
DEFAULT_FORMAT = "wav"
FORMAT_MIME: dict[str, str] = {"wav": "audio/wav"}


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str = DEFAULT_VOICE
    response_format: str = DEFAULT_FORMAT
    instructions: str | None = None
    language: str | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    model: str | None = None  # accepted + ignored for OpenAI-client compatibility

    model_config = {"protected_namespaces": ()}


def encode_wav(s: Synthesis) -> bytes:
    """16-bit PCM RIFF/WAVE, mono, at the adapter's sample rate."""
    samples = np.asarray(s.samples, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, samples, s.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def cache_key(
    model_version: str,
    voice: str,
    instructions: str | None,
    language: str | None,
    fmt: str,
    speed: float | None,
    text: str,
) -> str:
    """sha256 over every generation input (design doc §6.3). OmniVoice is
    non-deterministic per call, so this key -- not the audio -- is the stable
    identity of a request."""
    h = hashlib.sha256()
    speed_s = "" if speed is None else f"{speed:.3f}"
    h.update(
        f"{model_version}\0{voice}\0{instructions or ''}\0{language or ''}\0{fmt}\0{speed_s}\0"
        .encode()
    )
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def build_router(cfg: Config, adapter: TTSAdapter, loader: ModelLoader) -> APIRouter:
    router = APIRouter()
    slots = asyncio.Semaphore(cfg.slots)

    @router.post("/v1/audio/speech")
    async def speech(req: SpeechRequest) -> Response:
        text = req.input
        if len(text) > cfg.max_input_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"input is {len(text)} chars; max is {cfg.max_input_chars} "
                    "(CIS_MAX_INPUT_CHARS). Split long text into multiple requests."
                ),
            )
        fmt = req.response_format
        if fmt not in FORMAT_MIME:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported format '{fmt}'; this server serves "
                    f"{', '.join(FORMAT_MIME)} only (v1: no opus/mp3 transcoding)"
                ),
            )
        voices = adapter.voices()
        if req.voice not in voices:
            raise HTTPException(
                status_code=400,
                detail=f"unknown voice '{req.voice}'; see GET /info for the voice list",
            )

        try:
            await loader.get()
        except Exception as e:  # noqa: BLE001 -- surfaced as 503, load retried next call
            raise HTTPException(
                status_code=503, detail=f"model failed to load: {type(e).__name__}: {e}"
            ) from e

        sreq = SynthesisRequest(
            text=text,
            voice=req.voice,
            instructions=req.instructions,
            language=req.language,
            speed=req.speed,
        )
        async with slots:
            try:
                result = await asyncio.to_thread(adapter.synthesize, sreq)
            except InvalidRequestError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:  # noqa: BLE001
                logger.exception("synthesis failed")
                raise HTTPException(
                    status_code=500, detail=f"synthesis failed: {type(e).__name__}: {e}"
                ) from e

        data = encode_wav(result)
        seconds = round(len(result.samples) / result.sample_rate, 3)
        mv = adapter.model_version()
        key = cache_key(mv, req.voice, req.instructions, req.language, fmt, req.speed, text)
        headers = {
            "X-TTS-Cache-Hit": "0",
            "X-TTS-Duration-Seconds": str(seconds),
            "X-TTS-Chars": str(len(text)),
            "X-TTS-Model-Version": mv,
            "X-TTS-Cache-Key": key,
        }
        return Response(content=data, media_type=FORMAT_MIME[fmt], headers=headers)

    return router


def info_fields(cfg: Config, adapter: TTSAdapter) -> dict:
    """Task-level additions to ``/info``."""
    return {
        "voices": adapter.voices(),
        "default_voice": DEFAULT_VOICE,
        "default_format": DEFAULT_FORMAT,
        "formats": list(FORMAT_MIME),
        "capacity": {"slots": cfg.slots, "max_input_chars": cfg.max_input_chars},
    }
