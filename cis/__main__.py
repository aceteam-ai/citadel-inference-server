"""``python -m cis`` / ``cis``: validate config, then serve with uvicorn."""

from __future__ import annotations

import logging
import sys

from cis.config import Config
from cis.errors import ConfigError


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = Config.from_env()
        from cis.app import create_app

        app = create_app(cfg)
    except ConfigError as e:
        print(f"cis: configuration error: {e}", file=sys.stderr)
        return 2

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="info")  # noqa: S104
    return 0


if __name__ == "__main__":
    sys.exit(main())
