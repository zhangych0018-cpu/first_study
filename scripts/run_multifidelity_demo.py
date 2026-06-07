"""本脚本用于演示多保真校正模块的基本用法，展示如何把低保真模拟数据和高保真校准数据组合起来估计偏差。它主要用于说明接口形态和校准流程。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.analysis.plots import plot_multifidelity_calibration
from sws_bo.optimization.initial_design import generate_hybrid_design
from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
from sws_bo.surrogate.multifidelity import calibrate_with_measurements, estimate_bias, fit_autoregressive_multifidelity, predict_high_fidelity
from sws_bo.utils.io import save_json
from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "results" / "multifidelity_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    low_sim = MockDSGCSTSimulator(seed=10, noise_scale=1.0)
    high_sim = MockDSGCSTSimulator(seed=99, noise_scale=0.3)

    low_design = generate_hybrid_design(40, seed=10, normalized=False, problem=DSGSWSProblem).valid
    low_results = [low_sim.run(x) for x in low_design]
    success_mask = np.array([result.success for result in low_results], dtype=bool)
    low_design = low_design[success_mask]
    low_Y = np.array(
        [[result.Kc_mean, result.vp_std, result.ohmic_loss_mean, result.S11_max] for result in np.array(low_results, dtype=object)[success_mask]],
        dtype=float,
    )

    measurement_X = low_design[:8].copy()
    measurement_rows = []
    successful_measurements = []
    for x in measurement_X:
        res = high_sim.run(x)
        if not res.success:
            continue
        successful_measurements.append(x)
        measurement_rows.append(
            {
                **{name: float(x[idx]) for idx, name in enumerate(DSGSWSProblem.param_names)},
                "Kc_mean": res.Kc_mean * 0.97,
                "vp_std": res.vp_std * 1.06,
                "ohmic_loss_mean": res.ohmic_loss_mean * 1.03,
                "S11_max": res.S11_max + 0.8,
            }
        )
    measurement_X = np.asarray(successful_measurements, dtype=float)
    measurement_df = pd.DataFrame(measurement_rows)
    measurement_df.to_csv(output_dir / "mock_measurements.csv", index=False, encoding="utf-8")

    model = fit_autoregressive_multifidelity(
        low_design,
        low_Y,
        measurement_X,
        measurement_df[DSGSWSProblem.target_names].to_numpy(),
        DSGSWSProblem.target_names,
    )
    corrected = predict_high_fidelity(model, measurement_X, DSGSWSProblem.target_names)
    bias = estimate_bias(model)
    save_json(bias, output_dir / "bias_summary.json")

    calibration_df = pd.DataFrame(
        {
            "measured": measurement_df["Kc_mean"].to_numpy(),
            "corrected": corrected[:, 0],
        }
    )
    plot_multifidelity_calibration(calibration_df, output_dir / "multifidelity_calibration.png")
    print(output_dir)


if __name__ == "__main__":
    main()
