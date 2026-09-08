"""POST /v1/audio/speech: request shape, receipt headers, validation."""

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

from cis.adapters.tts import cache_key
from tests.test_app import wait_loaded

RECEIPT_HEADERS = [
    "X-TTS-Model-Version",
    "X-TTS-Cache-Key",
    "X-TTS-Chars",
    "X-TTS-Duration-Seconds",
    "X-TTS-Cache-Hit",
]


def test_speech_wav_bytes_and_receipt_headers(client):
    wait_loaded(client)
    r = client.post("/v1/audio/speech", json={"input": "Hello there"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
    for h in RECEIPT_HEADERS:
        assert h in r.headers, h
    assert r.headers["X-TTS-Chars"] == "11"
    assert r.headers["X-TTS-Cache-Hit"] == "0"
    assert r.headers["X-TTS-Model-Version"] == "stub-0.0.0+stub/test-model"
    assert len(r.headers["X-TTS-Cache-Key"]) == 64
    assert float(r.headers["X-TTS-Duration-Seconds"]) == 0.25
    samples, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000
    assert samples.ndim == 1
    assert len(samples) == 6000
    info = sf.info(io.BytesIO(r.content))
    assert info.subtype == "PCM_16" and info.channels == 1


def test_defaults_are_auto_and_wav(client):
    wait_loaded(client)
    r = client.post("/v1/audio/speech", json={"input": "x"})
    assert r.status_code == 200
    # Same key as an explicit auto/wav request => defaults resolved identically.
    explicit = client.post(
        "/v1/audio/speech", json={"input": "x", "voice": "auto", "response_format": "wav"}
    )
    assert explicit.headers["X-TTS-Cache-Key"] == r.headers["X-TTS-Cache-Key"]


def test_kokoro_shaped_request_is_accepted(client):
    """A caller that sends kokoro's full SpeechRequest (speed + model) must not 422."""
    wait_loaded(client)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "x", "voice": "auto", "response_format": "wav",
              "speed": 1.0, "model": "tts-1"},
    )
    assert r.status_code == 200


def test_named_preset_voice_and_instructions(client):
    wait_loaded(client)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "x", "voice": "stub-a", "instructions": "female, british accent"},
    )
    assert r.status_code == 200


def test_unknown_voice_400_points_at_info(client):
    wait_loaded(client)
    r = client.post("/v1/audio/speech", json={"input": "x", "voice": "a warm female voice"})
    assert r.status_code == 400
    assert "unknown voice" in r.json()["detail"]
    assert "/info" in r.json()["detail"]
    voices = client.get("/info").json()["voices"]
    assert "a warm female voice" not in voices


def test_non_wav_format_400(client):
    wait_loaded(client)
    for fmt in ("opus", "mp3", "flac", "WAV"):
        r = client.post("/v1/audio/speech", json={"input": "x", "response_format": fmt})
        assert r.status_code == 400, fmt
        assert f"unsupported format '{fmt}'" in r.json()["detail"]


def test_over_max_input_chars_413(stub_app_factory):
    client, _adapter, _ = stub_app_factory(CIS_MAX_INPUT_CHARS="10")
    wait_loaded(client)
    r = client.post("/v1/audio/speech", json={"input": "x" * 11})
    assert r.status_code == 413
    assert "CIS_MAX_INPUT_CHARS" in r.json()["detail"]
    assert client.post("/v1/audio/speech", json={"input": "x" * 10}).status_code == 200


def test_adapter_invalid_request_maps_to_400(client):
    wait_loaded(client)
    r = client.post(
        "/v1/audio/speech", json={"input": "x", "instructions": "invalid: purple accent"}
    )
    assert r.status_code == 400
    assert "Unsupported instruct" in r.json()["detail"]


def test_missing_input_422(client):
    assert client.post("/v1/audio/speech", json={"voice": "auto"}).status_code == 422


def test_speed_bounds_422(client):
    assert client.post("/v1/audio/speech", json={"input": "x", "speed": 3.0}).status_code == 422
    assert client.post("/v1/audio/speech", json={"input": "x", "speed": 0.1}).status_code == 422


def test_cache_key_covers_every_generation_input():
    base = dict(model_version="m", voice="auto", instructions=None, language=None,
                fmt="wav", speed=None, text="t")
    k0 = cache_key(**base)
    assert k0 == cache_key(**base)
    for field, val in [("model_version", "m2"), ("voice", "female"),
                       ("instructions", "male"), ("language", "en"),
                       ("speed", 1.2), ("text", "t2")]:
        assert cache_key(**{**base, field: val}) != k0, field


def test_lazy_load_failure_is_503_and_retries(stub_app_factory):
    client, adapter, fatal = stub_app_factory(CIS_PRELOAD="false", extra={"fail_load": True})
    r = client.post("/v1/audio/speech", json={"input": "x"})
    assert r.status_code == 503
    assert "fail_load" in r.json()["detail"]
    health = client.get("/health").json()
    assert health["status"] == "loading" and health["model_loaded"] is False
    assert "fail_load" in health["last_error"]
    assert fatal == [], "a LAZY load failure is never fatal"
    # Operator fixes the problem; the next request retries the load.
    adapter.fail_load = False
    assert client.post("/v1/audio/speech", json={"input": "x"}).status_code == 200
    assert adapter.load_calls == 2


def test_single_flight_load_under_concurrent_first_requests(stub_app_factory):
    client, adapter, _ = stub_app_factory(CIS_PRELOAD="false", CIS_SLOTS="8")
    adapter.load_gate = threading.Event()
    n = 6

    def hit():
        return client.post("/v1/audio/speech", json={"input": "x"})

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(hit) for _ in range(n)]
        # All N are now parked on the single in-flight load.
        for _ in range(200):
            if adapter.load_calls >= 1:
                break
            threading.Event().wait(0.01)
        threading.Event().wait(0.05)
        adapter.load_gate.set()
        results = [f.result(timeout=10) for f in futs]

    assert all(r.status_code == 200 for r in results)
    assert adapter.load_calls == 1, "N concurrent first requests must trigger ONE load"
    assert adapter.synth_calls == n


def test_semaphore_bounds_concurrent_synthesis(stub_app_factory):
    client, adapter, _ = stub_app_factory(CIS_SLOTS="1", extra={"synth_delay_s": 0.05})
    wait_loaded(client)
    n = 5
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(
            lambda _: client.post("/v1/audio/speech", json={"input": "x"}), range(n)
        ))
    assert all(r.status_code == 200 for r in results)
    assert adapter.synth_calls == n
    assert adapter.max_concurrent_synth == 1


def test_semaphore_allows_up_to_slots(stub_app_factory):
    client, adapter, _ = stub_app_factory(CIS_SLOTS="3", extra={"synth_delay_s": 0.1})
    wait_loaded(client)
    n = 6
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(
            lambda _: client.post("/v1/audio/speech", json={"input": "x"}), range(n)
        ))
    assert all(r.status_code == 200 for r in results)
    assert 1 <= adapter.max_concurrent_synth <= 3


def test_error_body_shape_is_fastapi_detail(client):
    """kokoro emits {"detail": ...}; the Go handler surfaces the body verbatim."""
    wait_loaded(client)
    r = client.post("/v1/audio/speech", json={"input": "x", "response_format": "opus"})
    assert set(json.loads(r.content)) == {"detail"}
