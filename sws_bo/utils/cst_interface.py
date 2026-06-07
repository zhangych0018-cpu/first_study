"""本模块封装真实 CST COM/Python 后端的调用细节，包括运行目录、重试、超时、结果解析和失败记录。它的目标是把真实仿真风险隔离在优化主流程之外。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import os
import shutil
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..data_schema import SimulationResult
from ..forward_model import ForwardSimulator
from ..problems.dsg_bwo_problem import DSGSWSProblem
from .io import ensure_dir, save_json
from .cst_result_export import DEFAULT_FILENAMES, export_standard_dsg_results
from .postprocessing import (
    PostprocessingError,
    parse_dsg_cst_results,
    parse_dsg_sparameter_results,
    validate_required_post_files,
)


class CSTBackendUnavailableError(RuntimeError):
    """当当前机器无法运行真实 CST 后端时抛出此异常，用于清晰区分环境缺失与普通业务错误。"""


def ensure_cst_python_paths(cst_amd64_dir: str | Path | None = None) -> Path:
    """定位 CST Python 库并注入 `sys.path`，让 BO 后端可以直接使用 `cst.interface` 与 `cst.results`。"""

    candidates: list[Path] = []
    if cst_amd64_dir:
        candidates.append(Path(cst_amd64_dir))
    env_text = os.environ.get("CST_AMD64_DIR")
    if env_text:
        candidates.append(Path(env_text))
    candidates.extend(
        [
            Path(r"D:\Programs\CST\AMD64"),
            Path(r"C:\Program Files\CST Studio Suite 2024\AMD64"),
            Path(r"C:\Program Files\CST Studio Suite 2025\AMD64"),
        ]
    )

    for amd64_dir in candidates:
        lib_dir = amd64_dir / "python_cst_libraries"
        if amd64_dir.exists() and lib_dir.exists():
            for entry in (str(lib_dir), str(amd64_dir)):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return amd64_dir

    raise CSTBackendUnavailableError(
        "CST 后端不可用：无法定位 CST Python 库目录，请设置 CST_AMD64_DIR 或在配置中提供 cst.amd64_dir。"
    )


def load_cst_modules(cst_amd64_dir: str | Path | None = None) -> tuple[Any, Any]:
    """导入 CST 官方 Python 模块；失败时包装成后端不可用错误，便于 BO 自动降级或记录。"""

    try:
        ensure_cst_python_paths(cst_amd64_dir)
        import cst.interface  # type: ignore
        import cst.results  # type: ignore

        return cst.interface, cst.results
    except CSTBackendUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - 依赖本机 CST 安装
        raise CSTBackendUnavailableError(f"CST 后端不可用：导入 CST Python 模块失败：{exc}") from exc


class CSTSimulator(ForwardSimulator):
    """封装真实 CST 自动化与导出解析流程，把运行目录、重试、超时和失败记录统一隐藏在一个后端类中。"""

    def __init__(
        self,
        template_path: str | Path,
        results_dir: str | Path,
        timeout: float = 3600.0,
        retries: int = 1,
        retry_backoff: float = 5.0,
        working_band_ghz: tuple[float, float] = (96.0, 110.0),
        postprocess_filenames: dict[str, str] | None = None,
        result_tree_items: dict[str, str] | None = None,
        parameter_mapping: dict[str, str] | None = None,
        fixed_parameters: dict[str, float | int | str] | None = None,
        cst_amd64_dir: str | Path | None = None,
        poll_seconds: float = 5.0,
        output_mode: str = "full_dsg",
    ) -> None:
        self.template_path = Path(template_path) if template_path else Path("__missing_template__.cst")
        self.results_dir = ensure_dir(results_dir)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_backoff = float(retry_backoff)
        self.working_band_ghz = tuple(float(v) for v in working_band_ghz)
        self.postprocess_filenames = {**DEFAULT_FILENAMES, **(postprocess_filenames or {})}
        if output_mode not in {"full_dsg", "sparameter_only"}:
            raise ValueError(f"Unknown CST output_mode: {output_mode}")
        self.output_mode = output_mode
        self.result_file_keys = (
            ["sparameters"]
            if self.output_mode == "sparameter_only"
            else ["dispersion_tm21", "dispersion_fundamental", "kc_tm21", "kc_fundamental", "sparameters"]
        )
        self.result_tree_items = result_tree_items or {}
        self.parameter_mapping = parameter_mapping or {name: name for name in DSGSWSProblem.param_names}
        self.fixed_parameters = fixed_parameters or {}
        self.poll_seconds = float(poll_seconds)
        self._cst_interface, self._cst_results = load_cst_modules(cst_amd64_dir)

    def run(self, x: dict | np.ndarray) -> SimulationResult:
        """执行一次完整的真实 CST 评估，并在成功时返回标准仿真结果，在失败时返回带诊断信息的失败结果。"""

        start = time.perf_counter()
        params = self._normalize_input(x)
        run_id = self._make_run_id()
        run_dir = ensure_dir(self.results_dir / run_id)
        raw_files = self._prepare_run_artifacts(run_dir, params)

        if not self.template_path.exists():
            return self._failure_result(
                start_time=start,
                failure_reason=f"template_missing:{self.template_path}",
                run_id=run_id,
                params=params,
                raw_files=raw_files,
            )

        last_error: Exception | None = None
        total_attempts = max(1, self.retries + 1)
        for attempt in range(1, total_attempts + 1):
            attempt_start = time.perf_counter()
            self._append_log(
                raw_files["stdout_log"],
                f"[attempt {attempt}] starting CST evaluation in {run_dir}\n",
            )
            try:  # pragma: no cover - 真实 CST 执行依赖本机环境和许可证状态
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._execute_single_attempt,
                        params=params,
                        run_dir=run_dir,
                        raw_files=raw_files,
                        attempt=attempt,
                    )
                    try:
                        future.result(timeout=self.timeout)
                    except FutureTimeoutError as exc:
                        raise TimeoutError(
                            f"CST run exceeded timeout {self.timeout:.1f}s after attempt {attempt}"
                        ) from exc
                validate_required_post_files({key: raw_files[key] for key in self.result_file_keys})
                metrics = self._parse_results(raw_files, params)
                self._append_log(raw_files["stdout_log"], f"[attempt {attempt}] postprocessing succeeded\n")
                return SimulationResult(
                    Kc_mean=metrics["Kc_mean"],
                    vp_std=metrics["vp_std"],
                    ohmic_loss_mean=metrics["ohmic_loss_mean"],
                    S11_max=metrics["S11_max"],
                    S21_mean=metrics.get("S21_mean"),
                    bandwidth_estimate=metrics.get("bandwidth_estimate"),
                    sim_time=time.perf_counter() - start,
                    success=True,
                    raw_files={key: str(value) for key, value in raw_files.items()},
                    extra_outputs={
                        key: value
                        for key, value in metrics.items()
                        if key
                        not in {
                            "Kc_mean",
                            "vp_std",
                            "ohmic_loss_mean",
                            "S11_max",
                            "S21_mean",
                            "bandwidth_estimate",
                        }
                    },
                    metadata={
                        "backend": "cst",
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "attempts_used": attempt,
                        "params": params,
                        "postprocessing_summary": {
                            "fc_low": metrics.get("fc_low"),
                            "fc_high": metrics.get("fc_high"),
                            "dispersion_flatness": metrics.get("dispersion_flatness"),
                        },
                    },
                )
            except Exception as exc:  # pragma: no cover - 这一分支主要通过失败处理路径间接覆盖
                last_error = exc
                self._append_exception_logs(exc, raw_files, attempt)
                if attempt < total_attempts:
                    time.sleep(self.retry_backoff)

        failure_reason = self._classify_failure(last_error, raw_files)
        return self._failure_result(
            start_time=start,
            failure_reason=failure_reason,
            run_id=run_id,
            params=params,
            raw_files=raw_files,
            attempts_used=total_attempts,
        )

    def _normalize_input(self, x: dict | np.ndarray) -> dict[str, float]:
        if isinstance(x, dict):
            base = {name: float(value) for name, value in x.items()}
            if all(name in base for name in DSGSWSProblem.param_names):
                return {name: base[name] for name in DSGSWSProblem.param_names}
            return base
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] == DSGSWSProblem.dim:
            params = DSGSWSProblem.as_dict(arr)
        else:
            names = DSGSWSProblem.param_names if arr.shape[-1] == DSGSWSProblem.dim else [f"x_{idx}" for idx in range(arr.shape[-1])]
            params = {name: float(arr[idx]) for idx, name in enumerate(names)}
        return {name: float(value) for name, value in params.items()}

    def _make_run_id(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"cst_run_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _prepare_run_artifacts(self, run_dir: Path, params: dict[str, float]) -> dict[str, Path]:
        raw_files = {
            "params_json": run_dir / "params.json",
            "stdout_log": run_dir / "cst_stdout.log",
            "stderr_log": run_dir / "cst_stderr.log",
            "com_error_log": run_dir / "cst_com_error.log",
        }
        for key, filename in self.postprocess_filenames.items():
            raw_files[key] = run_dir / filename
        cst_params = self._build_cst_parameter_payload(params)
        save_json({"optimization_params": params, "cst_params": cst_params}, raw_files["params_json"])
        self._append_log(raw_files["stdout_log"], f"template_path={self.template_path}\n")
        self._append_log(raw_files["stdout_log"], f"params={json.dumps(cst_params, ensure_ascii=False)}\n")
        return raw_files

    def _build_cst_parameter_payload(self, params: dict[str, float]) -> dict[str, float | int | str]:
        """把优化参数名映射为 CST 模板参数名，并叠加固定工程参数。"""

        payload: dict[str, float | int | str] = {}
        for source_name, target_name in self.parameter_mapping.items():
            if source_name in params:
                payload[target_name] = params[source_name]
        payload.update(self.fixed_parameters)
        return payload

    def _parse_results(self, raw_files: dict[str, Path], params: dict[str, float]) -> dict[str, Any]:
        if self.output_mode == "sparameter_only":
            return parse_dsg_sparameter_results(
                sparameters_path=raw_files["sparameters"],
                working_band_ghz=self.working_band_ghz,
            )
        return parse_dsg_cst_results(
            dispersion_tm21_path=raw_files["dispersion_tm21"],
            dispersion_fundamental_path=raw_files["dispersion_fundamental"],
            kc_tm21_path=raw_files["kc_tm21"],
            kc_fundamental_path=raw_files["kc_fundamental"],
            sparameters_path=raw_files["sparameters"],
            mode_frequencies_path=raw_files.get("mode_frequencies"),
            working_band_ghz=self.working_band_ghz,
            target_frequency_ghz=float(params.get("target_frequency_ghz", DSGSWSProblem.target_frequency_ghz)),
            beam_voltage_kv=float(params.get("beam_voltage_kv", 5.45)),
        )

    def _execute_single_attempt(
        self,
        *,
        params: dict[str, float],
        run_dir: Path,
        raw_files: dict[str, Path],
        attempt: int,
    ) -> None:
        """表示单次真实 CST 尝试的扩展点，实际工程可在这里填入打开模板、写参数、启动求解和导出结果的环境相关细节。"""

        _ = (params, run_dir, raw_files, attempt)
        run_project_path = self._copy_template_project(run_dir)
        cst_params = self._build_cst_parameter_payload(params)

        design_environment = None
        project = None
        try:
            design_environment = self._cst_interface.DesignEnvironment.connect_to_any_or_new()
            try:
                design_environment.set_quiet_mode(True)
            except Exception:
                pass
            project = design_environment.open_project(str(run_project_path))
            self._append_log(raw_files["stdout_log"], f"[attempt {attempt}] opened project={run_project_path}\n")

            self._apply_parameters(project, cst_params)
            self._save_project(project, run_project_path)
            self._run_solver(project)
            self._save_project(project, run_project_path)

            result_project = self._cst_results.ProjectFile(str(run_project_path), allow_interactive=True)
            results_3d = result_project.get_3d()
            tree_items = sorted(list(results_3d.get_tree_items()))
            (run_dir / "tree_items.txt").write_text("\n".join(tree_items) + "\n", encoding="utf-8")
            export_summary = export_standard_dsg_results(
                results_3d=results_3d,
                tree_items=tree_items,
                run_dir=run_dir,
                filenames=self.postprocess_filenames,
                result_tree_items=self.result_tree_items,
            )
            save_json(export_summary, run_dir / "standard_dsg_exports.json")
            if self.output_mode == "full_dsg" and not export_summary["required_complete"]:
                raise PostprocessingError(f"真实 CST 标准导出不完整: {export_summary['missing']}")
            if self.output_mode == "sparameter_only" and "sparameters" not in export_summary["exported"]:
                raise PostprocessingError(f"真实 CST S 参数导出失败: {export_summary['missing'].get('sparameters')}")
        except Exception as exc:
            raw_files["stderr_log"].write_text(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                encoding="utf-8",
            )
            raise
        finally:
            self._close_quietly(project, design_environment)

    def _copy_template_project(self, run_dir: Path) -> Path:
        """复制 CST 工程文件及其同名展开目录，保证结果库和参数文件在独立 run 目录内保持完整。"""

        run_project_path = (run_dir / self.template_path.name).resolve()
        shutil.copy2(self.template_path, run_project_path)

        source_bundle_dir = self.template_path.with_suffix("")
        target_bundle_dir = run_project_path.with_suffix("")
        if source_bundle_dir.exists() and source_bundle_dir.is_dir():
            shutil.copytree(source_bundle_dir, target_bundle_dir, dirs_exist_ok=True)
        return run_project_path

    def _apply_parameters(self, project: Any, params: dict[str, float | int | str]) -> None:
        """把当前设计点写入 CST 参数表；优先使用 Python API，失败时回退到 History 中的 `StoreParameter`。"""

        model3d = project.model3d
        for name, value in params.items():
            value_text = str(value)
            try:
                model3d.store_parameter(name, value_text)
                continue
            except Exception:
                pass
            try:
                model3d.add_to_history(f"set parameter {name}", f'StoreParameter "{name}", "{value_text}"')
            except Exception as exc:
                raise RuntimeError(f"写入 CST 参数失败: {name}={value_text}") from exc

        # 参数化几何通常需要 rebuild 才能刷新模型；不同 CST 版本暴露的方法名不完全一致，所以采用多路尝试。
        for method_name in ("rebuild", "Rebuild"):
            method = getattr(model3d, method_name, None)
            if callable(method):
                try:
                    method()
                    return
                except Exception:
                    pass
        try:
            model3d.add_to_history("rebuild after BO parameter update", "Rebuild")
        except Exception:
            pass

    def _run_solver(self, project: Any) -> None:
        """运行当前 CST 工程活动求解器，并兼容阻塞和非阻塞两类 CST Python 行为。"""

        model3d = project.model3d
        model3d.run_solver()
        while True:
            try:
                running = bool(model3d.is_solver_running())
            except Exception:
                running = False
            if not running:
                return
            time.sleep(max(0.1, self.poll_seconds))

    def _save_project(self, project: Any, project_path: Path) -> None:
        for args, kwargs in (
            ((project_path,), {"allow_overwrite": True}),
            ((str(project_path),), {"allow_overwrite": True}),
            ((), {}),
        ):
            try:
                project.save(*args, **kwargs)
                return
            except Exception:
                continue

    def _close_quietly(self, project: Any | None, design_environment: Any | None) -> None:
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
        if design_environment is not None:
            try:
                design_environment.close()
            except Exception:
                pass

    def _classify_failure(self, exc: Exception | None, raw_files: dict[str, Path]) -> str:
        if exc is None:
            return "cst_failed_unknown"
        if isinstance(exc, TimeoutError):
            return f"timeout:{exc}"
        if isinstance(exc, PostprocessingError):
            return f"postprocessing_error:{exc}"
        if isinstance(exc, NotImplementedError):
            return f"cst_backend_unavailable:{exc}"
        existing_exports = [name for name in self.result_file_keys if raw_files[name].exists()]
        if existing_exports and len(existing_exports) < len(self.result_file_keys):
            return f"partial_results:available={existing_exports}"
        return f"cst_runtime_error:{exc}"

    def _failure_result(
        self,
        *,
        start_time: float,
        failure_reason: str,
        run_id: str,
        params: dict[str, float],
        raw_files: dict[str, Path],
        attempts_used: int = 0,
    ) -> SimulationResult:
        return SimulationResult(
            Kc_mean=np.nan,
            vp_std=np.nan,
            ohmic_loss_mean=np.nan,
            S11_max=np.nan,
            sim_time=time.perf_counter() - start_time,
            success=False,
            failure_reason=failure_reason,
            raw_files={key: str(value) for key, value in raw_files.items()},
            metadata={
                "backend": "cst",
                "run_id": run_id,
                "run_dir": str(Path(raw_files["params_json"]).parent),
                "attempts_used": attempts_used,
                "params": params,
            },
        )

    def _append_exception_logs(self, exc: Exception, raw_files: dict[str, Path], attempt: int) -> None:
        message = f"[attempt {attempt}] {type(exc).__name__}: {exc}\n"
        self._append_log(raw_files["stderr_log"], message)
        self._append_log(raw_files["com_error_log"], message)

    def _append_log(self, path: Path, message: str) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message)
