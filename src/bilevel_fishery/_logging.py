"""Centralized logging configuration for the bilevel-fishery package."""

from __future__ import annotations

import logging
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: LogLevel = "INFO") -> None:
    """Configure the root logger with a consistent format across the package.

    Parameters
    ----------
    level
        Logging verbosity level. Defaults to ``"INFO"``.

    Notes
    -----
    Idempotent: calling it multiple times keeps the most recent configuration
    via ``force=True``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
