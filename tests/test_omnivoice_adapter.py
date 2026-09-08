"""OmniVoice adapter against a FAKE OmniVoice class + fake torch: the exact
kwargs mapping, and -- load-bearing for v1 -- the ABSENCE of every voice-
cloning / ASR key. No torch, no omnivoice, no download."""

from types import SimpleNamespace

import numpy as np
import pytest

from cis.adapters import omnivoice as ov
from cis.adapters.base import SynthesisRequest
from cis.errors import InvalidRequestError
from tests.conftest import make_config

CLONING_GENERATE_KEYS = {"ref_audio", "ref_text", "voice_clone_prompt"}
ASR_LOAD_KEYS = {"load_asr", "asr_model_name", "asr_device"}


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.generate_calls: list[dict] = []
        self.raise_value_error: str | None = None

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.raise_value_error:
            raise ValueError(self.raise_value_error)
        return [np.zeros(2400, dtype=np.float32)]


class FakeOmniVoice:
    instances: list[FakeModel] = []
    from_pretrained_calls: list[tuple] = []

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        cls.from_pretrained_calls.append((path, args, kwargs))
        m = FakeModel()
        cls.instances.append(m)
        return m


def fake_torch(cuda_available: bool, cuda_build: bool = True):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=SimpleNamespace(cuda="12.8" if cuda_build else None),
        float16="torch.float16",
        bfloat16="torch.bfloat16",
        float32="torch.float32",
    )


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeOmniVoice.instances = []
    FakeOmniVoice.from_pretrained_calls = []


def make_adapter(cuda=True, ensure_calls=None, **cfg_overrides):
    base = {"CIS_ADAPTER": "omnivoice", "CIS_MODEL": "k2-fsa/OmniVoice", "CIS_DEVICE": "auto"}
    cfg = make_config(**{**base, **cfg_overrides})
    ensure_calls = [] if ensure_calls is None else ensure_calls

    def ensure(model, **kw):
        ensure_calls.append((model, kw))
        return "/hub/models--k2-fsa--OmniVoice/snapshots/abc"

    return OmniVoiceAdapterFactory(cfg, cuda, ensure), ensure_calls


def OmniVoiceAdapterFactory(cfg, cuda, ensure):
    return ov.OmniVoiceAdapter(
        cfg,
        omnivoice_cls_fn=lambda: FakeOmniVoice,
        torch_fn=lambda: fake_torch(cuda),
        ensure_snapshot_fn=ensure,
    )


def test_load_maps_device_dtype_and_passes_local_path_no_asr():
    adapter, ensure_calls = make_adapter(cuda=True, CIS_DTYPE="float16",
                                         CIS_MODEL_REVISION="v1", HF_TOKEN="tok")
    adapter.load()
    assert ensure_calls == [("k2-fsa/OmniVoice", dict(revision="v1", token="tok",
                                                       disk_preflight=True))]
    (path, args, kwargs), = FakeOmniVoice.from_pretrained_calls
    assert path == "/hub/models--k2-fsa--OmniVoice/snapshots/abc", "local dir, not repo id"
    assert args == ()
    assert kwargs == {"device_map": "cuda", "dtype": "torch.float16"}, (
        "dtype= (not torch_dtype=), device_map=, and NOTHING else"
    )
    assert not (ASR_LOAD_KEYS & set(kwargs))
    assert adapter.describe()["device"] == "cuda"
    assert adapter.describe()["dtype"] == "float16"
    assert adapter.describe()["sample_rate"] == 24000
    assert adapter.describe()["voice_cloning"] is False


def test_auto_device_and_dtype_fall_back_to_cpu_fp32():
    adapter, _ = make_adapter(cuda=False)
    adapter.load()
    (_, _, kwargs), = FakeOmniVoice.from_pretrained_calls
    assert kwargs == {"device_map": "cpu", "dtype": "torch.float32"}


def test_explicit_cuda_without_gpu_is_a_hard_error():
    adapter, _ = make_adapter(cuda=False, CIS_DEVICE="cuda")
    with pytest.raises(RuntimeError, match="no GPU is visible"):
        adapter.load()
    assert FakeOmniVoice.from_pretrained_calls == []
    with pytest.raises(RuntimeError, match="no CUDA support"):
        ov.resolve_device(fake_torch(False, cuda_build=False), "cuda:0")


def test_generate_kwargs_auto_voice_has_no_instruct():
    adapter, _ = make_adapter()
    kw = adapter.generate_kwargs(SynthesisRequest(text="hi"))
    assert kw == {"text": "hi", "num_step": 32}
    assert "instruct" not in kw


def test_generate_kwargs_preset_instructions_override_language_speed():
    adapter, _ = make_adapter(extra={"num_step": 16})
    kw = adapter.generate_kwargs(SynthesisRequest(text="hi", voice="female-british"))
    assert kw["instruct"] == "female, british accent" and kw["num_step"] == 16
    kw = adapter.generate_kwargs(
        SynthesisRequest(text="hi", voice="female-british", instructions="male, low pitch",
                         language="en", speed=1.25)
    )
    assert kw == {"text": "hi", "num_step": 16, "instruct": "male, low pitch",
                  "language": "en", "speed": 1.25}
    # Whitespace-only instructions do not override the preset.
    kw = adapter.generate_kwargs(SynthesisRequest(text="hi", voice="male", instructions="  "))
    assert kw["instruct"] == "male"


def test_synthesize_never_passes_cloning_keys_and_returns_first_audio():
    adapter, _ = make_adapter()
    adapter.load()
    out = adapter.synthesize(SynthesisRequest(text="hello", voice="whisper", language="en"))
    (call,) = FakeOmniVoice.instances[0].generate_calls
    assert call == {"text": "hello", "num_step": 32, "instruct": "whisper", "language": "en"}
    assert not (CLONING_GENERATE_KEYS & set(call)), "voice cloning is v2"
    assert out.sample_rate == 24000
    assert out.samples.dtype == np.float32 and out.samples.shape == (2400,)


def test_upstream_instruct_value_error_becomes_invalid_request():
    adapter, _ = make_adapter()
    adapter.load()
    FakeOmniVoice.instances[0].raise_value_error = (
        "Unsupported instruct items found in purple: 'purple' -> 'purple' (unsupported)"
    )
    with pytest.raises(InvalidRequestError, match="Unsupported instruct"):
        adapter.synthesize(SynthesisRequest(text="x", instructions="purple"))


def test_synthesize_before_load_is_an_error():
    adapter, _ = make_adapter()
    with pytest.raises(RuntimeError, match="before load"):
        adapter.synthesize(SynthesisRequest(text="x"))


def test_presets_are_closed_and_composed_of_upstream_vocabulary():
    """Every preset must be made only of items omnivoice's _resolve_instruct
    accepts (mirrored here from omnivoice 0.2.1's _INSTRUCT_CATEGORIES), or it
    would 400 at generate time."""
    valid = {
        "male", "female",
        "child", "teenager", "young adult", "middle-aged", "elderly",
        "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
        "whisper",
        "american accent", "british accent", "australian accent", "chinese accent",
        "canadian accent", "indian accent", "korean accent", "portuguese accent",
        "russian accent", "japanese accent",
    }
    adapter, _ = make_adapter()
    voices = adapter.voices()
    assert voices[0] == "auto"
    assert set(voices[1:]) == set(ov.PRESETS)
    assert len(voices) == len(set(voices))
    for name, instruct in ov.PRESETS.items():
        items = [i.strip() for i in instruct.split(",")]
        assert all(i in valid for i in items), (name, instruct)
        assert ", ".join(items) == instruct, "comma + space separator, as upstream requires"


def test_model_version_and_license():
    adapter, _ = make_adapter()
    assert adapter.model_version() == f"omnivoice-{ov.omnivoice_package_version()}+k2-fsa/OmniVoice"
    assert adapter.model_license() == "CC-BY-NC", "the card's literal; no version invented"
    other, _ = make_adapter(CIS_MODEL="someone/finetune")
    assert other.model_license() == "unknown"
    over, _ = make_adapter(CIS_MODEL="someone/finetune", extra={"model_license": "MIT"})
    assert over.model_license() == "MIT"


def test_bad_num_step_rejected_at_construction():
    with pytest.raises(ValueError, match="num_step"):
        make_adapter(extra={"num_step": "lots"})
    with pytest.raises(ValueError, match="num_step"):
        make_adapter(extra={"num_step": 0})
