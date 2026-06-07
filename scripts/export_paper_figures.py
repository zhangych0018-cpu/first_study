"""本脚本面向论文或汇报材料的出图需求，负责将项目已有分析结果整理成命名清晰、便于引用的图文件。它强调复现性和批量导出，而不是交互式分析。"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    source_dir = PROJECT_ROOT / "data" / "results" / "dsg_mock_demo" / "figures"
    target_dir = PROJECT_ROOT / "data" / "results" / "paper_figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "hypervolume_history.png": "fig1_pipeline.png",
        "pareto_3d.png": "fig2_pareto_3d.png",
        "feasibility_history.png": "fig3_constraint_feasibility.png",
        "pareto_2d.png": "fig4_pareto_2d.png",
    }
    for src_name, dst_name in mapping.items():
        src = source_dir / src_name
        if src.exists():
            shutil.copyfile(src, target_dir / dst_name)
    print(target_dir)


if __name__ == "__main__":
    main()
