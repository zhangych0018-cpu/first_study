"""本测试模块用于验证 DSG 专用 mock BO 流程能否顺利完成最小轮次运行，并检查关键输出文件与状态字段。"""

from pathlib import Path


def test_dsg_mock_bo_loop_runs(tmp_path: Path):
    import pandas as pd

    from sws_bo.optimization.bo_loop import SWSBayesianOptimizer
    from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem

    optimizer = SWSBayesianOptimizer(
        backend="mock_dsg",
        problem=DSGSWSProblem,
        n_initial=8,
        n_iterations=2,
        batch_size=1,
        seed=17,
        output_dir=tmp_path / "dsg_mock_run",
    )
    summary = optimizer.run()
    evaluations = pd.read_csv(tmp_path / "dsg_mock_run" / "evaluations.csv")
    assert summary["n_evaluated"] >= 8
    assert {"W", "P", "T", "G", "H"}.issubset(set(evaluations.columns))
    assert "Kc_TM21_mean" in evaluations.columns
