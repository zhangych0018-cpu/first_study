"""本模块封装项目中常用的文件系统读写辅助函数，帮助各个脚本和优化流程以统一方式创建目录、保存 JSON 或 CSV 等结构化结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回对应路径对象，是项目各类写盘操作最常用的基础工具。"""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """以 UTF-8 编码把 Python 对象保存为 JSON 文件，统一项目中的结构化结果写盘行为。"""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path_obj


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件并返回解析后的 Python 对象。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """把 DataFrame 保存为 CSV 文件，作为表格类结果的统一输出接口。"""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    df.to_csv(path_obj, index=False, encoding="utf-8")
    return path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件并返回配置字典。"""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """把配置字典写回 YAML 文件，便于导出或缓存实验设置。"""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path_obj
