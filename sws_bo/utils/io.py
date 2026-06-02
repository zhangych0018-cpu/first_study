"""Shared file IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON with UTF-8 encoding."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path_obj


def load_json(path: str | Path) -> dict[str, Any]:
    """Read JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    df.to_csv(path_obj, index=False, encoding="utf-8")
    return path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read YAML config."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write YAML config."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path_obj
