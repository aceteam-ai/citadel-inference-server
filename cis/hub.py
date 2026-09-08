"""HuggingFace cache placement + self-provisioning.

``ensure_snapshot`` resolves ``CIS_MODEL`` to a LOCAL directory the adapter
can hand to ``from_pretrained``:

1. an existing local path is used as-is;
2. an already-cached repo (``local_files_only=True``) is returned with no
   network call and no preflight -- the citadel-cli ``MODEL_CACHE_PULL``
   pre-fetch writes the same hub layout into the same mounted directory, so a
   pre-pulled node pulls nothing here;
3. otherwise the disk preflight runs, then ``snapshot_download``.

Two deliberate choices, both learned the hard way in the diffusers sidecar
(see its ``model_preflight.py`` docstrings):

* **Only ``HF_HOME`` is honored, never ``cache_dir=``.** ``HF_HUB_CACHE``
  defaults to ``HF_HOME/hub`` -- the layout citadel-cli's cache tables and
  ``MODEL_CACHE_PULL`` agree on. Passing ``cache_dir=HF_HOME`` would write
  weights where nothing else looks.
* **Revision is applied HERE, not via ``from_pretrained``.** OmniVoice's
  ``from_pretrained`` resolves a repo id with a bare ``snapshot_download(name)``
  (no revision kwarg), so ``CIS_MODEL_REVISION`` can only take effect if we
  download first and pass the resolved local path (its ``os.path.isdir``
  branch).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from cis import preflight

logger = logging.getLogger("cis.hub")

SnapshotDownloadFn = Callable[..., str]


def hub_cache_dir() -> str:
    """Where ``snapshot_download`` will write, per huggingface_hub's own
    resolution (``HF_HUB_CACHE`` -> ``HF_HOME/hub``)."""
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def default_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision, token=token, local_files_only=local_files_only
    )


def ensure_snapshot(
    model: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    disk_preflight: bool = True,
    snapshot_download_fn: SnapshotDownloadFn = default_snapshot_download,
    cache_dir_fn: Callable[[], str] = hub_cache_dir,
    run_preflight_fn: Callable[..., int | None] = preflight.run_preflight,
) -> str:
    """Return a local directory for ``model`` (see module docstring). Raises
    ``InsufficientDiskSpaceError`` on a confirmed disk shortfall; download
    errors propagate as-is."""
    if os.path.isdir(model):
        logger.info("model %s is a local directory; using it directly", model)
        return model

    try:
        path = snapshot_download_fn(
            model, revision=revision, token=token, local_files_only=True
        )
        logger.info("model %s already cached at %s; no download", model, path)
        return path
    except Exception:  # noqa: BLE001 -- LocalEntryNotFoundError or any cache miss
        pass

    cache_dir = cache_dir_fn()
    if disk_preflight:
        run_preflight_fn(model, cache_dir, revision=revision, token=token)
    else:
        logger.info("disk preflight disabled (CIS_DISK_PREFLIGHT=false)")

    logger.info("downloading %s (revision=%s) into %s", model, revision or "main", cache_dir)
    path = snapshot_download_fn(model, revision=revision, token=token, local_files_only=False)
    logger.info("downloaded %s to %s", model, path)
    return path
