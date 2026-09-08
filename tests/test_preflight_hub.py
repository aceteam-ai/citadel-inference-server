"""Disk preflight (pure decision + fail-open/fail-closed) and ensure_snapshot."""

import pytest

from cis import hub, preflight
from cis.errors import InsufficientDiskSpaceError
from cis.preflight import RepoFileInfo, plan_disk_preflight, run_preflight

GIB = 1024**3


def test_plan_fits():
    assert plan_disk_preflight("/c", 1 * GIB, 4 * GIB, margin_bytes=2 * GIB) is None


def test_plan_shortfall_refuses_with_message():
    with pytest.raises(InsufficientDiskSpaceError) as ei:
        plan_disk_preflight("/c", 3 * GIB, 4 * GIB, margin_bytes=2 * GIB)
    msg = str(ei.value)
    assert "insufficient disk space at /c" in msg
    assert "5.0 GiB" in msg and "3.0 GiB" in msg and "4.0 GiB" in msg
    assert "downloading nothing" in msg


def test_plan_margin_is_counted():
    plan_disk_preflight("/c", 2 * GIB, 2 * GIB, margin_bytes=0)
    with pytest.raises(InsufficientDiskSpaceError):
        plan_disk_preflight("/c", 2 * GIB, 2 * GIB, margin_bytes=1)


def _tree(*sizes):
    def list_fn(repo, *, revision=None, token=None):
        return [RepoFileInfo(path=f"f{i}", size=s) for i, s in enumerate(sizes)] + [
            RepoFileInfo(path="dir", size=0, is_file=False)
        ]

    return list_fn


def test_run_preflight_sums_files_and_fits():
    got = run_preflight(
        "org/repo", "/cache",
        list_repo_files_fn=_tree(GIB, GIB),
        available_disk_bytes_fn=lambda p: 10 * GIB,
    )
    assert got == 2 * GIB


def test_run_preflight_confirmed_shortfall_fails_closed():
    with pytest.raises(InsufficientDiskSpaceError):
        run_preflight(
            "org/repo", "/cache",
            list_repo_files_fn=_tree(3 * GIB),
            available_disk_bytes_fn=lambda p: 4 * GIB,
        )


def test_run_preflight_fails_open_on_estimate_error():
    def boom(repo, *, revision=None, token=None):
        raise ConnectionError("hf down")

    assert run_preflight(
        "org/repo", "/cache", list_repo_files_fn=boom, available_disk_bytes_fn=lambda p: 0
    ) is None


def test_run_preflight_fails_open_on_probe_error():
    def boom(p):
        raise OSError("statfs")

    assert run_preflight(
        "org/repo", "/cache", list_repo_files_fn=_tree(GIB), available_disk_bytes_fn=boom
    ) is None


def test_nearest_existing_dir(tmp_path):
    assert preflight.nearest_existing_dir(str(tmp_path / "a" / "b")) == str(tmp_path)


# --- ensure_snapshot ---------------------------------------------------------


class FakeSnapshot:
    def __init__(self, cached: bool):
        self.cached = cached
        self.calls: list[dict] = []

    def __call__(self, repo, *, revision=None, token=None, local_files_only=False):
        self.calls.append(dict(repo=repo, revision=revision, token=token,
                               local_files_only=local_files_only))
        if local_files_only and not self.cached:
            raise FileNotFoundError("not cached")
        return f"/hub/models--{repo.replace('/', '--')}/snapshots/{revision or 'main'}"


def test_ensure_snapshot_local_dir_used_directly(tmp_path):
    snap = FakeSnapshot(cached=False)
    assert hub.ensure_snapshot(str(tmp_path), snapshot_download_fn=snap) == str(tmp_path)
    assert snap.calls == []


def test_ensure_snapshot_cached_skips_download_and_preflight():
    snap = FakeSnapshot(cached=True)
    ran = []
    path = hub.ensure_snapshot(
        "k2-fsa/OmniVoice", revision="abc", token="tok",
        snapshot_download_fn=snap, run_preflight_fn=lambda *a, **k: ran.append(1),
        cache_dir_fn=lambda: "/hub",
    )
    assert path.endswith("/snapshots/abc")
    assert snap.calls == [dict(repo="k2-fsa/OmniVoice", revision="abc", token="tok",
                               local_files_only=True)]
    assert ran == []


def test_ensure_snapshot_downloads_after_preflight_with_revision():
    snap = FakeSnapshot(cached=False)
    ran = []

    def pre(repo, cache_dir, *, revision=None, token=None):
        ran.append((repo, cache_dir, revision, token))
        return 1

    path = hub.ensure_snapshot(
        "k2-fsa/OmniVoice", revision="v1", token="t",
        snapshot_download_fn=snap, run_preflight_fn=pre, cache_dir_fn=lambda: "/hub",
    )
    assert path.endswith("/snapshots/v1")
    assert ran == [("k2-fsa/OmniVoice", "/hub", "v1", "t")]
    assert [c["local_files_only"] for c in snap.calls] == [True, False]
    assert snap.calls[-1]["revision"] == "v1", "CIS_MODEL_REVISION must reach the download"


def test_ensure_snapshot_preflight_refusal_downloads_nothing():
    snap = FakeSnapshot(cached=False)

    def pre(*a, **k):
        raise InsufficientDiskSpaceError("no room")

    with pytest.raises(InsufficientDiskSpaceError):
        hub.ensure_snapshot("org/repo", snapshot_download_fn=snap, run_preflight_fn=pre,
                            cache_dir_fn=lambda: "/hub")
    assert [c["local_files_only"] for c in snap.calls] == [True]


def test_ensure_snapshot_preflight_can_be_disabled():
    snap = FakeSnapshot(cached=False)
    ran = []
    hub.ensure_snapshot("org/repo", disk_preflight=False, snapshot_download_fn=snap,
                        run_preflight_fn=lambda *a, **k: ran.append(1),
                        cache_dir_fn=lambda: "/hub")
    assert ran == []
    assert snap.calls[-1]["local_files_only"] is False


def test_hub_cache_dir_is_hf_home_hub(tmp_path):
    """Only HF_HOME is honored; the hub cache is HF_HOME/hub -- the layout
    citadel-cli's MODEL_CACHE_PULL writes. Never pass cache_dir=HF_HOME.
    Run in a subprocess because huggingface_hub resolves its constants at
    import time."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
    env["HF_HOME"] = str(tmp_path)
    out = subprocess.check_output(
        [sys.executable, "-c", "from cis import hub; print(hub.hub_cache_dir())"],
        env=env, text=True,
    ).strip()
    assert out == str(tmp_path / "hub")
