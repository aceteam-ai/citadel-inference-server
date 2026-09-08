import pytest

from cis import adapters
from cis.adapters.omnivoice import OmniVoiceAdapter
from cis.adapters.stub import StubAdapter
from cis.errors import ConfigError
from tests.conftest import make_config


def test_registry_families():
    assert adapters.families() == {"omnivoice": "tts", "stub": "tts"}


def test_create_picks_family_from_cis_adapter():
    assert isinstance(adapters.create(make_config(CIS_ADAPTER="stub")), StubAdapter)
    a = adapters.create(make_config(CIS_ADAPTER="omnivoice", CIS_MODEL="k2-fsa/OmniVoice"))
    assert isinstance(a, OmniVoiceAdapter)


def test_unknown_family_is_config_error():
    with pytest.raises(ConfigError, match="not a known adapter family"):
        adapters.create(make_config(CIS_ADAPTER="kokoro"))


def test_task_mismatch_is_config_error():
    adapters.register("fake-image", "image", lambda cfg: None)
    try:
        with pytest.raises(ConfigError, match="implements task 'image'"):
            adapters.create(make_config(CIS_ADAPTER="fake-image"))
    finally:
        adapters._FAMILIES.pop("fake-image")


def test_adapter_construction_error_is_config_error():
    with pytest.raises(ConfigError, match="num_step"):
        adapters.create(make_config(CIS_ADAPTER="omnivoice", extra={"num_step": -1}))


def test_omnivoice_adapter_module_imports_without_torch():
    """`import cis.adapters.omnivoice` must stay torch-free so the hermetic
    suite and the image smoke never need the heavy layer at import time."""
    import subprocess
    import sys

    code = (
        "import sys; import cis.adapters.omnivoice; "
        "assert 'torch' not in sys.modules and 'omnivoice' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", code])
