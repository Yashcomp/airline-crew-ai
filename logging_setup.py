from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> int:
    """Configure root logging for the app and CLI tools.

    Level resolution order: explicit argument, then AIRLINE_LOG_LEVEL env var,
    then INFO. Any invalid level name falls back to INFO. Returns the numeric
    level applied so callers can inspect it.
    """
    if level is None:
        level = os.getenv("AIRLINE_LOG_LEVEL", "INFO").upper()
    numeric = getattr(logging, level, None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    return numeric
