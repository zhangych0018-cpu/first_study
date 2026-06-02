"""Windows COM wrapper for the real CST backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import time
import uuid
from pathlib import Path

import numpy as np

from ..data_schema import SimulationResult
from ..forward_model import ForwardSimulator
from ..problems.dsg_bwo_problem import DSGSWSProblem
from .io import ensure_dir, save_json
from .postprocessing import (
    PostprocessingError,
    parse_dsg_cst_results,
    validate_required_post_files,
)


class CSTBackendUnavailableError(RuntimeError):
    """Raised when the current machine cannot run the CST backend."""


class CSTSimulator(ForwardSimulator):
    """Isolated wrapper around CST COM automation and export parsing."""

    def __init__(
        self,
        template_path: str | Path,
        results_dir: str | Path,
        timeout: float = 3600.0,
        retries: int = 1,
        retry_backoff: float = 5.0,
        working_band_ghz: tuple[float, float] = (96.0, 110.0),
        postprocess_filenames: dict[str, str] | None = None,
    ) -> None:
        self.template_path = Path(template_path) if template_path else Path("__missing_template__.cst")
        self.results_dir = ensure_dir(results_dir)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_backoff = float(retry_backoff)
        self.working_band_ghz = tuple(float(v) for v in working_band_ghz)
        self.postprocess_filenames = postprocess_filenames or {
            "dispersion_tm21": "dispersion_tm21.txt",
            "dispersion_fundamental": "dispersion_fundamental.txt",
            "kc_tm21": "kc_tm21.txt",
            "kc_fundamental": "kc_fundamental.txt",
            "sparameters": "sparameters.txt",
        }
        self.result_file_keys = list(self.postprocess_filenames.keys())
        try:
            import win32com.client as win32_client
        except Exception as exc:  # pragma: no cover - depends on Windows COM
            raise CSTBackendUnavailableError(
                "CST backend unavailable: win32com or CST Studio Suite is not installed."
            ) from exc
        self._win32_client = win32_client

    def run(self, x: dict | np.ndarray) -> SimulationResult:
        """Run one CST evaluation while preserving full diagnostics."""

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
            try:  # pragma: no cover - real CST execution is environment-dependent
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
            except Exception as exc:  # pragma: no cover - mainly exercised through failure handling
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
            return {name: float(value) for name, value in x.items()}
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
        save_json(params, raw_files["params_json"])
        self._append_log(raw_files["stdout_log"], f"template_path={self.template_path}\n")
        self._append_log(raw_files["stdout_log"], f"params={json.dumps(params, ensure_ascii=False)}\n")
        return raw_files

    def _parse_results(self, raw_files: dict[str, Path], params: dict[str, float]) -> dict[str, float]:
        return parse_dsg_cst_results(
            dispersion_tm21_path=raw_files["dispersion_tm21"],
            dispersion_fundamental_path=raw_files["dispersion_fundamental"],
            kc_tm21_path=raw_files["kc_tm21"],
            kc_fundamental_path=raw_files["kc_fundamental"],
            sparameters_path=raw_files["sparameters"],
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
        """Run one actual CST attempt.

        The real COM interaction is intentionally isolated here. Teams can replace
        this body with environment-specific project loading, parameter injection,
        solver launch, export, and cleanup logic without touching the BO loop.
        """

        _ = (params, run_dir, raw_files, attempt)
        raise NotImplementedError(
            "Real CST automation requires a local parameterized template and export setup."
        )

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
