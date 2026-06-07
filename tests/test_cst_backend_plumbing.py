"""本测试模块验证真实 CST 后端的工程化胶水逻辑。

这些测试不启动 CST，也不依赖许可证；它们只覆盖容易导致真实链路失败的本地文件与命令行参数处理：
1. `.cst` 文件必须连同同名展开目录一起复制；
2. 命令行显式传入的 `0` 必须被当成有效覆盖值，而不是回退到配置默认值。
"""

from __future__ import annotations

from pathlib import Path

from scripts.run_bo import choose_cli_or_config
from sws_bo.utils.cst_interface import CSTSimulator


def test_choose_cli_or_config_keeps_zero_values():
    assert choose_cli_or_config(0, 5, 9) == 0
    assert choose_cli_or_config(None, 5, 9) == 5
    assert choose_cli_or_config(None, None, 9) == 9


def test_copy_template_project_includes_unpacked_cst_directory(tmp_path: Path):
    template = tmp_path / "DSG_SWS.cst"
    template.write_text("placeholder cst", encoding="utf-8")
    unpacked = tmp_path / "DSG_SWS"
    (unpacked / "Model").mkdir(parents=True)
    (unpacked / "Model" / "Parameters.json").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    simulator = CSTSimulator.__new__(CSTSimulator)
    simulator.template_path = template

    copied_project = simulator._copy_template_project(run_dir)

    assert copied_project == (run_dir / "DSG_SWS.cst").resolve()
    assert copied_project.exists()
    assert (run_dir / "DSG_SWS" / "Model" / "Parameters.json").exists()
