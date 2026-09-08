import pytest

from cis.config import Config
from cis.errors import ConfigError
from tests.conftest import BASE_ENV, make_config


def test_pinned_contract_vars_parse():
    cfg = Config.from_env(
        {
            "CIS_TASK": "tts",
            "CIS_ADAPTER": "omnivoice",
            "CIS_MODEL": "k2-fsa/OmniVoice",
            "CIS_DEVICE": "cuda",
            "CIS_DTYPE": "float16",
            "CIS_SLOTS": "3",
            "CIS_MAX_INPUT_CHARS": "1234",
            "CIS_PRELOAD": "false",
            "CIS_EXTRA": '{"num_step": 16}',
        }
    )
    assert cfg.task == "tts"
    assert cfg.adapter == "omnivoice"
    assert cfg.model == "k2-fsa/OmniVoice"
    assert cfg.device == "cuda"
    assert cfg.dtype == "float16"
    assert cfg.slots == 3
    assert cfg.max_input_chars == 1234
    assert cfg.preload is False
    assert cfg.extra == {"num_step": 16}


def test_defaults():
    cfg = make_config()
    assert cfg.port == 8000, "container port is pinned to :8000 by the cross-phase contract"
    assert cfg.hf_home == "/root/.cache/huggingface"
    assert cfg.slots == 2
    assert cfg.max_input_chars == 5000
    assert cfg.preload is True
    assert cfg.disk_preflight is True
    assert cfg.dtype == "auto"
    assert cfg.revision is None
    assert cfg.extra == {}


@pytest.mark.parametrize("missing", ["CIS_TASK", "CIS_ADAPTER", "CIS_MODEL"])
def test_required_vars(missing):
    env = dict(BASE_ENV)
    env[missing] = ""
    with pytest.raises(ConfigError, match=missing):
        Config.from_env(env)


def test_bad_extra_json_refuses_startup():
    with pytest.raises(ConfigError, match="CIS_EXTRA is not valid JSON"):
        Config.from_env({**BASE_ENV, "CIS_EXTRA": '{"num_step": 16'})


def test_extra_must_be_object():
    with pytest.raises(ConfigError, match="JSON object"):
        Config.from_env({**BASE_ENV, "CIS_EXTRA": "[1,2]"})


@pytest.mark.parametrize(
    "var,val",
    [("CIS_SLOTS", "0"), ("CIS_SLOTS", "x"), ("CIS_MAX_INPUT_CHARS", "-1"), ("PORT", "0")],
)
def test_bad_ints(var, val):
    with pytest.raises(ConfigError, match=var):
        Config.from_env({**BASE_ENV, var: val})


def test_bad_bool():
    with pytest.raises(ConfigError, match="CIS_PRELOAD"):
        Config.from_env({**BASE_ENV, "CIS_PRELOAD": "maybe"})


def test_bad_task_dtype_device():
    with pytest.raises(ConfigError, match="CIS_TASK"):
        Config.from_env({**BASE_ENV, "CIS_TASK": "image"})
    with pytest.raises(ConfigError, match="CIS_DTYPE"):
        Config.from_env({**BASE_ENV, "CIS_DTYPE": "int8"})
    with pytest.raises(ConfigError, match="CIS_DEVICE"):
        Config.from_env({**BASE_ENV, "CIS_DEVICE": "tpu"})
    assert Config.from_env({**BASE_ENV, "CIS_DEVICE": "cuda:1"}).device == "cuda:1"


def test_hf_token_aliases():
    assert Config.from_env({**BASE_ENV, "HF_TOKEN": "a"}).hf_token == "a"
    assert Config.from_env({**BASE_ENV, "HUGGING_FACE_HUB_TOKEN": "b"}).hf_token == "b"
    assert Config.from_env(BASE_ENV).hf_token is None
