"""Exceptions shared between the runtime core and the adapters."""


class ConfigError(ValueError):
    """The CIS_* environment is missing or malformed. Raised at startup so the
    container exits with a clear message instead of serving a half-configured
    app."""


class InvalidRequestError(ValueError):
    """An adapter rejected a request on its own (family-specific) grounds --
    e.g. OmniVoice's closed instruct vocabulary. The task layer maps this to a
    400 carrying the adapter's message verbatim, so the caller sees the
    upstream explanation (which for OmniVoice already lists the valid items)."""


class InsufficientDiskSpaceError(RuntimeError):
    """The disk preflight confirmed the download cannot fit. Fails CLOSED:
    nothing is downloaded."""
