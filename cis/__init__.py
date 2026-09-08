"""citadel-inference-server (CIS): a generic, config-driven inference server.

Base runtime (config, HF cache, single-flight model load, /health, /info,
request semaphore, receipt headers, disk preflight) plus thin per-family
adapters under ``cis.adapters``. Design of record:
aceteam-ai/citadel-cli ``docs/design-generic-inference-server.md``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("citadel-inference-server")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
