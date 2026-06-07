"""本脚本是 DSG 贝叶斯优化的通用命令行入口，负责加载配置、选择后端、启动优化循环并保存运行结果。无论是 mock 还是真实 CST，这里都是用户最常用的调度入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sws_bo.optimization.bo_loop import SWSBayesianOptimizer
from sws_bo.problems import resolve_problem
from sws_bo.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWS Bayesian optimization")
    parser.add_argument("--backend", choices=["mock_dsg", "cst"], default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pf-min", type=float, default=None)
    parser.add_argument("--cst-template", type=str, default=None)
    parser.add_argument("--cst-results-dir", type=str, default=None)
    parser.add_argument("--cst-timeout", type=float, default=None)
    parser.add_argument("--cst-retries", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(PROJECT_ROOT / args.config)
    opt_cfg = cfg.get("optimization", {})
    cst_cfg = cfg.get("cst", {})
    problem_cfg = cfg.get("problem", {})
    problem = resolve_problem(problem_cfg.get("name", "dsg"))
    post_cfg = cst_cfg.get("postprocessing", {})
    working_band = tuple(post_cfg.get("working_band_ghz", [96.0, 110.0]))
    postprocess_filenames = {
        "dispersion_tm21": post_cfg.get("dispersion_tm21_filename", "dispersion_tm21.txt"),
        "dispersion_fundamental": post_cfg.get("dispersion_fundamental_filename", "dispersion_fundamental.txt"),
        "kc_tm21": post_cfg.get("kc_tm21_filename", "kc_tm21.txt"),
        "kc_fundamental": post_cfg.get("kc_fundamental_filename", "kc_fundamental.txt"),
        "sparameters": post_cfg.get("sparameters_filename", "sparameters.txt"),
        "mode_frequencies": post_cfg.get("mode_frequencies_filename", "mode_frequencies.csv"),
    }
    optimizer = SWSBayesianOptimizer(
        backend=args.backend or opt_cfg.get("backend", "mock_dsg"),
        problem=problem,
        n_initial=args.n_initial or opt_cfg.get("n_initial", 20),
        n_iterations=args.n_iterations or opt_cfg.get("n_iterations", 5),
        batch_size=args.batch_size or opt_cfg.get("batch_size", 2),
        seed=args.seed or opt_cfg.get("seed", 42),
        pf_min=args.pf_min or opt_cfg.get("pf_min", 0.8),
        output_dir=opt_cfg.get("output_dir", "data/results/dsg_default_run"),
        resume=args.resume,
        ard=opt_cfg.get("ard", True),
        cst_template_path=args.cst_template if args.cst_template is not None else cst_cfg.get("template_path"),
        cst_results_dir=args.cst_results_dir if args.cst_results_dir is not None else cst_cfg.get("results_dir"),
        cst_timeout=args.cst_timeout if args.cst_timeout is not None else cst_cfg.get("timeout_per_sim", 3600),
        cst_retries=args.cst_retries if args.cst_retries is not None else cst_cfg.get("retries", 1),
        cst_retry_backoff=cst_cfg.get("retry_backoff_seconds", 60.0),
        cst_working_band_ghz=working_band,
        cst_postprocess_filenames=postprocess_filenames,
        cst_result_tree_items=post_cfg.get("result_tree_items", {}),
        cst_parameter_mapping=cst_cfg.get("parameter_mapping", {}),
        cst_fixed_parameters=cst_cfg.get("fixed_parameters", {}),
        cst_amd64_dir=cst_cfg.get("amd64_dir"),
        cst_poll_seconds=cst_cfg.get("poll_seconds", 5.0),
        cst_output_mode=cst_cfg.get("output_mode", "full_dsg"),
    )
    summary = optimizer.run()
    print(summary)


if __name__ == "__main__":
    main()
