"""本模块提供轻量级日志工具，用于在不引入复杂依赖的前提下统一项目运行过程中的文本记录格式。它主要服务脚本和主循环的可追踪性。"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(name: str = "sws_bo", log_file: str | Path | None = None) -> logging.Logger:
    """创建一个带控制台输出、可选文件输出的轻量日志器，供脚本和主循环统一记录运行过程。"""

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
