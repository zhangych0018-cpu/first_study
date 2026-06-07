"""本模块实现 DSG 贝叶斯优化主循环，负责协调采样、仿真、代理训练、候选推荐、失败容错和结果持久化。它是整个项目的执行中枢。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..acquisition.constrained_qnehvi import recommend_candidates
from ..acquisition.constraints import feasible_pareto_mask
from ..analysis.hypervolume import compute_hypervolume
from ..analysis.pareto import select_representative_designs
from ..data_schema import SimulationResult
from ..geometry.dsg_sws import validate_dsg_geometry
from ..problems import DSGSWSProblem, resolve_problem
from ..surrogate.train import train_independent_gp
from ..utils.cst_interface import CSTBackendUnavailableError, CSTSimulator
from ..utils.io import ensure_dir, load_json, save_dataframe, save_json
from ..utils.logging import configure_logger
from ..utils.mock_dsg_cst import MockDSGCSTSimulator
from .initial_design import generate_hybrid_design


@dataclass
class BOIterationSummary:
    """记录每一轮贝叶斯优化的摘要信息，例如候选数、成功数和关键评价指标，便于回顾运行历史。"""

    iteration: int
    n_evaluated: int
    feasible_rate: float
    hypervolume: float


class SWSBayesianOptimizer:
    """实现对 DSG 慢波结构友好的多目标贝叶斯优化主循环，协调采样、仿真、建模和结果持久化。"""

    def __init__(
        self,
        backend: str = "mock_dsg",
        problem=DSGSWSProblem,
        n_initial: int = 20,
        n_iterations: int = 5,
        batch_size: int = 2,
        seed: int = 42,
        pf_min: float = 0.8,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
        ard: bool = True,
        cst_template_path: str | Path | None = None,
        cst_results_dir: str | Path | None = None,
        cst_timeout: float = 3600.0,
        cst_retries: int = 1,
        cst_retry_backoff: float = 60.0,
        cst_working_band_ghz: tuple[float, float] = (96.0, 110.0),
        cst_postprocess_filenames: dict[str, str] | None = None,
        cst_result_tree_items: dict[str, str] | None = None,
        cst_parameter_mapping: dict[str, str] | None = None,
        cst_fixed_parameters: dict[str, float | int | str] | None = None,
        cst_amd64_dir: str | Path | None = None,
        cst_poll_seconds: float = 5.0,
    ) -> None:
        self.problem = resolve_problem(problem)
        self.problem_name = getattr(self.problem, "name", self.problem.__name__)
        self.backend = backend
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed
        self.pf_min = pf_min
        self.ard = ard
        self.output_dir = Path(output_dir or Path("data") / "results" / f"{backend}_run")
        ensure_dir(self.output_dir)
        self.logger = configure_logger(log_file=self.output_dir / "run.log")
        self.resume_path = Path(resume) if resume else None
        self.simulator = self._build_simulator(
            cst_template_path,
            cst_results_dir,
            cst_timeout,
            cst_retries,
            cst_retry_backoff,
            cst_working_band_ghz,
            cst_postprocess_filenames,
            cst_result_tree_items,
            cst_parameter_mapping,
            cst_fixed_parameters,
            cst_amd64_dir,
            cst_poll_seconds,
        )
        self.evaluations = pd.DataFrame()
        self.history = pd.DataFrame()
        self.model = None
        if self.resume_path is not None and self.resume_path.exists():
            self._load_checkpoint(self.resume_path)

    def _build_simulator(
        self,
        cst_template_path,
        cst_results_dir,
        cst_timeout,
        cst_retries,
        cst_retry_backoff,
        cst_working_band_ghz,
        cst_postprocess_filenames,
        cst_result_tree_items,
        cst_parameter_mapping,
        cst_fixed_parameters,
        cst_amd64_dir,
        cst_poll_seconds,
    ):
        if self.backend == "mock_dsg":
            return MockDSGCSTSimulator(seed=self.seed)
        try:
            return CSTSimulator(
                cst_template_path,
                cst_results_dir or self.output_dir / "cst_exports",
                timeout=cst_timeout,
                retries=cst_retries,
                retry_backoff=cst_retry_backoff,
                working_band_ghz=cst_working_band_ghz,
                postprocess_filenames=cst_postprocess_filenames,
                result_tree_items=cst_result_tree_items,
                parameter_mapping=cst_parameter_mapping,
                fixed_parameters=cst_fixed_parameters,
                cst_amd64_dir=cst_amd64_dir,
                poll_seconds=cst_poll_seconds,
            )
        except CSTBackendUnavailableError as exc:
            self.logger.warning("CST unavailable, falling back to mock backend: %s", exc)
            self.backend = "mock_dsg"
            return MockDSGCSTSimulator(seed=self.seed)

    def run(self) -> dict:
        """执行完整 BO 循环并导出结果产物，是脚本层调度 mock 或真实 CST 优化时的主入口。"""

        start = time.perf_counter()
        if self.evaluations.empty:
            self._initialize_dataset()
        for iteration in range(self._completed_iterations(), self.n_iterations):
            self.logger.info("Starting BO iteration %d", iteration + 1)
            success_df = self._successful_evaluations()
            if len(success_df) < 4:
                self.logger.warning("Too few successful points for GP fit, sampling random fallback batch.")
                candidate_phys = generate_hybrid_design(
                    self.batch_size,
                    seed=self.seed + 100 + iteration,
                    problem=self.problem,
                    validator=self._design_validator,
                    repair_fn=self._design_repair,
                ).valid
            else:
                self.model = self._train_surrogate(success_df)
                candidate_phys = self._recommend_physical_candidates(success_df)
            self._evaluate_and_record(candidate_phys, stage=f"bo_iter_{iteration+1}")
            self._update_history(iteration + 1)
            self._save_checkpoint(self.output_dir / "checkpoint_latest.json")

        total_time = time.perf_counter() - start
        summary = self._finalize(total_time)
        return summary

    def _completed_iterations(self) -> int:
        return 0 if self.history.empty else int(self.history["iteration"].max())

    def _initialize_dataset(self) -> None:
        design = generate_hybrid_design(
            self.n_initial,
            seed=self.seed,
            normalized=False,
            problem=self.problem,
            validator=self._design_validator,
            repair_fn=self._design_repair,
        )
        self._evaluate_and_record(design.valid, stage="initial")
        self._update_history(0)

    def _design_validator(self, point: np.ndarray, problem) -> bool:
        return bool(validate_dsg_geometry(point).is_valid and problem.validate_x(point))

    def _design_repair(self, point: np.ndarray, problem) -> np.ndarray | None:
        candidate = np.clip(np.asarray(point, dtype=float), problem.bounds[:, 0], problem.bounds[:, 1])
        if self._design_validator(candidate, problem):
            return candidate
        for blend in np.linspace(0.15, 0.55, 3):
            repaired = (1.0 - blend) * candidate + blend * problem.reference_design
            repaired = np.clip(repaired, problem.bounds[:, 0], problem.bounds[:, 1])
            if self._design_validator(repaired, problem):
                return repaired
        return None

    def _evaluate_and_record(self, points_phys: np.ndarray, stage: str) -> None:
        rows = []
        for point in np.asarray(points_phys, dtype=float):
            try:
                result = self.simulator.run(point)
            except Exception as exc:
                self.logger.exception("Simulator raised an exception for stage=%s point=%s", stage, point.tolist())
                result = SimulationResult(
                    Kc_mean=np.nan,
                    vp_std=np.nan,
                    ohmic_loss_mean=np.nan,
                    S11_max=np.nan,
                    success=False,
                    failure_reason=f"simulator_exception:{type(exc).__name__}:{exc}",
                    metadata={"stage": stage, "point": point.tolist()},
                )
            row = {name: float(point[idx]) for idx, name in enumerate(self.problem.param_names)}
            row.update(result.to_dict())
            row["stage"] = stage
            row["problem_name"] = self.problem_name
            row["is_feasible"] = bool(result.success and self.problem.check_constraint(result.S11_max))
            if result.success:
                row["neg_Kc_mean"] = -result.Kc_mean
            else:
                row["neg_Kc_mean"] = np.nan
            rows.append(row)
        batch_df = pd.DataFrame(rows)
        self.evaluations = pd.concat([self.evaluations, batch_df], ignore_index=True)

    def _successful_evaluations(self) -> pd.DataFrame:
        return self.evaluations[self.evaluations["success"]].reset_index(drop=True)

    def _train_surrogate(self, success_df: pd.DataFrame):
        X = torch.tensor(self.problem.normalize(success_df[self.problem.param_names].to_numpy()), dtype=torch.double)
        Y = torch.tensor(success_df[self.problem.target_names].to_numpy(), dtype=torch.double)
        return train_independent_gp(X, Y, ard=self.ard)

    def _recommend_physical_candidates(self, success_df: pd.DataFrame) -> np.ndarray:
        train_X = torch.tensor(self.problem.normalize(success_df[self.problem.param_names].to_numpy()), dtype=torch.double)
        train_Y = torch.tensor(success_df[self.problem.target_names].to_numpy(), dtype=torch.double)
        recommendation = recommend_candidates(
            model=self.model,
            train_X=train_X,
            train_Y_raw=train_Y,
            q=self.batch_size,
            pf_min=self.pf_min,
            problem=self.problem,
        )
        return recommendation.physical

    def _update_history(self, iteration: int) -> None:
        success_df = self._successful_evaluations()
        if len(success_df) == 0:
            hv = 0.0
            feasible_rate = 0.0
        else:
            objectives = success_df[["neg_Kc_mean", "vp_std", "ohmic_loss_mean"]].to_numpy()
            feasible = success_df["is_feasible"].to_numpy(dtype=bool)
            hv = compute_hypervolume(objectives[feasible], self.problem.hypervolume_ref_point, maximize=False) if feasible.any() else 0.0
            feasible_rate = float(feasible.mean())
        row = pd.DataFrame(
            [
                {
                    "iteration": iteration,
                    "n_evaluated": len(self.evaluations),
                    "feasible_rate": feasible_rate,
                    "hypervolume": hv,
                    "problem_name": self.problem_name,
                }
            ]
        )
        self.history = pd.concat([self.history, row], ignore_index=True)

    def _save_checkpoint(self, path: str | Path) -> None:
        save_json({"evaluations_file": str(self.output_dir / "evaluations.csv"), "history_file": str(self.output_dir / "history.csv")}, path)
        save_dataframe(self.evaluations, self.output_dir / "evaluations.csv")
        save_dataframe(self.history, self.output_dir / "history.csv")

    def _load_checkpoint(self, path: str | Path) -> None:
        checkpoint = load_json(path)
        eval_file = Path(checkpoint["evaluations_file"])
        hist_file = Path(checkpoint["history_file"])
        if eval_file.exists():
            self.evaluations = pd.read_csv(eval_file)
        if hist_file.exists():
            self.history = pd.read_csv(hist_file)

    def export_pareto_dataframe(self) -> pd.DataFrame:
        """从成功样本中整理出可行 Pareto 数据表，供分析脚本和最终候选筛选复用。"""

        success_df = self._successful_evaluations()
        if len(success_df) == 0:
            return success_df
        obj = success_df[["neg_Kc_mean", "vp_std", "ohmic_loss_mean"]].to_numpy()
        raw = success_df[self.problem.target_names].to_numpy()
        mask = feasible_pareto_mask(obj, raw, success_df["success"].to_numpy(dtype=bool), problem=self.problem)
        return success_df.loc[mask].reset_index(drop=True)

    def _finalize(self, total_time: float) -> dict:
        pareto_df = self.export_pareto_dataframe()
        representative = select_representative_designs(pareto_df) if len(pareto_df) else pareto_df
        save_dataframe(self.evaluations, self.output_dir / "evaluations.csv")
        save_dataframe(self.history, self.output_dir / "history.csv")
        save_dataframe(pareto_df, self.output_dir / "pareto.csv")
        if len(representative):
            save_dataframe(representative, self.output_dir / "representative_designs.csv")
        summary = {
            "backend": self.backend,
            "problem_name": self.problem_name,
            "n_evaluated": int(len(self.evaluations)),
            "n_successful": int(self.evaluations["success"].sum()) if len(self.evaluations) else 0,
            "feasible_rate": float(self.evaluations["is_feasible"].mean()) if len(self.evaluations) else 0.0,
            "best_hypervolume": float(self.history["hypervolume"].max()) if len(self.history) else 0.0,
            "wall_clock_time": total_time,
            "output_dir": str(self.output_dir),
        }
        save_json(summary, self.output_dir / "summary.json")
        return summary
