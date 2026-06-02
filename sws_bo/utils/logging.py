"""Small logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(name: str = "sws_bo", log_file: str | Path | None = None) -> logging.Logger:
    """Create a console logger with optional file sink."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
