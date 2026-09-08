"""Disk preflight before a self-provisioning download.

A port of the pure core of citadel-cli's ``services/diffusers-service/
model_preflight.py`` (itself a mirror of ``internal/jobs/disk_space.go``):
estimate the repo's byte size from HuggingFace's tree API, compare against
free space at the cache directory plus a safety margin, and REFUSE rather than
fill the disk.

Contract (same as the Go original):

* Fail OPEN on a metadata-fetch or disk-probe error (gated repo, network
  hiccup, API shape change): a pull proceeding un-preflighted beats blocking a
  legitimate download because OUR estimate could not be computed.
* Fail CLOSED (``InsufficientDiskSpaceError``) only on a CONFIRMED shortfall.

Unlike the diffusers sidecar, the estimate here is exact rather than an upper
bound: a ``from_pretrained``-style adapter (OmniVoice) loads the ENTIRE repo
(``model.safetensors`` + ``audio_tokenizer/`` + tokenizer files), so summing
the whole tree is precisely what will land on disk and a refusal is never the
false-positive citadel#913 had to soften on the diffusers path.

Every I/O function is injectable so the decision is unit-tested with no
network and no real filesystem probe.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from cis.errors import InsufficientDiskSpaceError

logger = logging.getLogger("cis.preflight")

# Mirrors diskSafetyMarginBytes in citadel-cli internal/jobs/disk_space.go.
DEFAULT_DISK_SAFETY_MARGIN_BYTES = 2 * 1024**3  # 2 GiB


@dataclass(frozen=True)
class RepoFileInfo:
    path: str
    size: int
    is_file: bool = True


ListRepoFilesFn = Callable[..., Iterable[RepoFileInfo]]
AvailableDiskBytesFn = Callable[[str], int]


def human_bytes(n: int) -> str:
    n = max(int(n), 0)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TiB"


def default_list_repo_files(
    repo_id: str, *, revision: str | None = None, token: str | None = None
) -> list[RepoFileInfo]:
    """Production backend: HF's tree API via ``list_repo_tree``, which resolves
    LFS pointers to their real byte size inline."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi(token=token)
    out: list[RepoFileInfo] = []
    for item in api.list_repo_tree(repo_id, recursive=True, revision=revision, token=token):
        if isinstance(item, RepoFile):
            out.append(RepoFileInfo(path=item.path, size=item.size or 0, is_file=True))
    return out


def estimate_repo_size_bytes(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    list_repo_files_fn: ListRepoFilesFn = default_list_repo_files,
) -> int:
    return sum(
        e.size for e in list_repo_files_fn(repo_id, revision=revision, token=token) if e.is_file
    )


def nearest_existing_dir(path: str) -> str:
    """Walk up until a directory exists, so a first-ever download (cache dir
    not yet created) can still be probed. Same caveat as the Go original: if
    the bind-mount point itself is missing this describes the container's
    overlay, not the host disk."""
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return path
        path = parent
    return path


def default_available_disk_bytes(path: str) -> int:
    return shutil.disk_usage(nearest_existing_dir(path)).free


def plan_disk_preflight(
    dir_: str,
    required_bytes: int,
    available_bytes: int,
    margin_bytes: int = DEFAULT_DISK_SAFETY_MARGIN_BYTES,
) -> None:
    """The pure decision. Returns None to proceed; raises on a shortfall."""
    required_bytes = max(required_bytes, 0)
    margin_bytes = max(margin_bytes, 0)
    needed = required_bytes + margin_bytes
    if available_bytes < needed:
        raise InsufficientDiskSpaceError(
            f"insufficient disk space at {dir_}: need {human_bytes(needed)} "
            f"({human_bytes(required_bytes)} estimated download + "
            f"{human_bytes(margin_bytes)} safety margin) but only "
            f"{human_bytes(available_bytes)} free -- downloading nothing"
        )


def run_preflight(
    repo_id: str,
    cache_dir: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    margin_bytes: int = DEFAULT_DISK_SAFETY_MARGIN_BYTES,
    list_repo_files_fn: ListRepoFilesFn = default_list_repo_files,
    available_disk_bytes_fn: AvailableDiskBytesFn = default_available_disk_bytes,
) -> int | None:
    """Estimate + probe + decide. Returns the estimated byte size when the
    check ran (fit or not -- a non-fit raises), or None when it was skipped
    because the estimate/probe itself failed (fail-open)."""
    try:
        required = estimate_repo_size_bytes(
            repo_id, revision=revision, token=token, list_repo_files_fn=list_repo_files_fn
        )
    except Exception as e:  # noqa: BLE001 -- fail open by contract
        logger.warning("disk preflight skipped: could not estimate size of %s: %s", repo_id, e)
        return None
    try:
        available = available_disk_bytes_fn(cache_dir)
    except Exception as e:  # noqa: BLE001 -- fail open by contract
        logger.warning("disk preflight skipped: could not probe free space at %s: %s",
                       cache_dir, e)
        return None
    plan_disk_preflight(cache_dir, required, available, margin_bytes)
    logger.info("disk preflight ok: %s needs %s, %s free at %s",
                repo_id, human_bytes(required), human_bytes(available), cache_dir)
    return required
