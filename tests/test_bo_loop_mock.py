"""本测试模块用于验证通用 BO 主循环在 mock 仿真环境下的基本健壮性，包括失败记录、数据更新和运行完成条件。"""

from pathlib import Path


def test_mock_bo_loop_runs(tmp_path: Path):
    from sws_bo.optimization.bo_loop import SWSBayesianOptimizer

    optimizer = SWSBayesianOptimizer(
        backend="mock_dsg",
        n_initial=8,
        n_iterations=2,
        batch_size=1,
        seed=5,
        output_dir=tmp_path / "dsg_mock_run_general",
    )
    summary = optimizer.run()
    assert summary["n_evaluated"] >= 8
    assert (tmp_path / "dsg_mock_run_general" / "summary.json").exists()


def test_failed_simulation_logged(tmp_path: Path):
    import pandas as pd

    from sws_bo.data_schema import SimulationResult
    from sws_bo.optimization.bo_loop import SWSBayesianOptimizer
    from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
    from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator

    class OneReturnedFailureSimulator:
        def __init__(self):
            self.mock = MockDSGCSTSimulator(seed=11)
            self.calls = 0

        def run(self, x):
            self.calls += 1
            if self.calls == 2:
                return SimulationResult(
                    Kc_mean=float("nan"),
                    vp_std=float("nan"),
                    ohmic_loss_mean=float("nan"),
                    S11_max=float("nan"),
                    success=False,
                    failure_reason="injected_returned_failure",
                )
            return self.mock.run(x)

    optimizer = SWSBayesianOptimizer(
        backend="mock_dsg",
        n_initial=5,
        n_iterations=1,
        batch_size=1,
        seed=7,
        output_dir=tmp_path / "dsg_failed_logged",
    )
    optimizer.simulator = OneReturnedFailureSimulator()
    summary = optimizer.run()
    evaluations = pd.read_csv(tmp_path / "dsg_failed_logged" / "evaluations.csv")
    failed_rows = evaluations[~evaluations["success"]]

    assert summary["n_evaluated"] == len(evaluations)
    assert len(failed_rows) >= 1
    assert failed_rows["failure_reason"].notna().all()
    assert failed_rows["failure_reason"].str.contains("injected_returned_failure").any()
    assert failed_rows["neg_Kc_mean"].isna().all()
    successful_rows = evaluations[evaluations["success"]]
    assert len(successful_rows) == int(summary["n_successful"])
    assert successful_rows[DSGSWSProblem.target_names].notna().all().all()


def test_bo_loop_survives_one_failed_simulation(tmp_path: Path):
    import pandas as pd

    from sws_bo.optimization.bo_loop import SWSBayesianOptimizer
    from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator

    class OneRaisedFailureSimulator:
        def __init__(self):
            self.mock = MockDSGCSTSimulator(seed=13)
            self.calls = 0

        def run(self, x):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("injected_runtime_failure")
            return self.mock.run(x)

    optimizer = SWSBayesianOptimizer(
        backend="mock_dsg",
        n_initial=5,
        n_iterations=1,
        batch_size=1,
        seed=9,
        output_dir=tmp_path / "dsg_survives_failure",
    )
    optimizer.simulator = OneRaisedFailureSimulator()
    summary = optimizer.run()
    evaluations = pd.read_csv(tmp_path / "dsg_survives_failure" / "evaluations.csv")

    assert summary["n_evaluated"] == 6
    assert (~evaluations["success"]).any()
    failed_row = evaluations.loc[~evaluations["success"]].iloc[0]
    assert "simulator_exception:RuntimeError:injected_runtime_failure" in failed_row["failure_reason"]
    assert int(summary["n_successful"]) == int(evaluations["success"].sum())
