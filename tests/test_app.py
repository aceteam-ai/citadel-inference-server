"""/health and /info: the shapes citadel-cli's handler and the fabric read."""

import threading
import time

from cis import __version__

PINNED_INFO_KEYS = {"model", "adapter", "task", "model_license", "voices"}


def wait_loaded(client, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").json()["model_loaded"]:
            return
        time.sleep(0.01)
    raise AssertionError("model never reported loaded")


def test_health_preload_reaches_up(stub_app_factory):
    client, adapter, fatal = stub_app_factory()
    wait_loaded(client)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "up"
    assert body["model_loaded"] is True
    assert body["model_version"] == "stub-0.0.0+stub/test-model"
    assert body["slots"] == 2
    assert adapter.load_calls == 1
    assert fatal == []


def test_health_lazy_mode_reports_loading_until_first_request(stub_app_factory):
    client, adapter, _ = stub_app_factory(CIS_PRELOAD="false")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "loading"
    assert r.json()["model_loaded"] is False
    assert adapter.load_calls == 0
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 200
    assert client.get("/health").json()["status"] == "up"
    assert adapter.load_calls == 1


def test_health_is_200_while_loading(stub_app_factory):
    client, adapter, _ = stub_app_factory(CIS_PRELOAD="false")
    adapter.load_gate = threading.Event()
    # Kick a lazy load off from a thread and poll health while it is blocked.
    t = threading.Thread(
        target=lambda: client.post("/v1/audio/speech", json={"input": "hi"}), daemon=True
    )
    t.start()
    for _ in range(50):
        r = client.get("/health")
        assert r.status_code == 200
        if adapter.load_calls:
            break
        time.sleep(0.01)
    assert r.json() == {
        "status": "loading",
        "model_loaded": False,
        "model_version": "stub-0.0.0+stub/test-model",
        "slots": 2,
        "device": "cpu",
    }
    adapter.load_gate.set()
    t.join(5)
    assert client.get("/health").json()["status"] == "up"


def test_info_pinned_shape(client):
    wait_loaded(client)
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert PINNED_INFO_KEYS <= set(body)
    assert body["model"] == "stub/test-model"
    assert body["adapter"] == "stub"
    assert body["task"] == "tts"
    assert isinstance(body["model_license"], str) and body["model_license"]
    assert isinstance(body["voices"], list)
    assert body["voices"][0] == "auto"
    assert all(isinstance(v, str) for v in body["voices"])
    # additive fields
    assert body["model_loaded"] is True
    assert body["default_voice"] == "auto"
    assert body["default_format"] == "wav"
    assert body["formats"] == ["wav"]
    assert body["capacity"] == {"slots": 2, "max_input_chars": 5000}
    assert body["cis_version"] == __version__
    assert body["load_seconds"] is not None


def test_info_available_before_load(stub_app_factory):
    client, _adapter, _ = stub_app_factory(CIS_PRELOAD="false")
    body = client.get("/info").json()
    assert PINNED_INFO_KEYS <= set(body)
    assert body["model_loaded"] is False
    assert body["load_seconds"] is None
