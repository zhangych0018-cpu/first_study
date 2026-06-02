"""Export paper-ready figure filenames from an analyzed run."""

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
