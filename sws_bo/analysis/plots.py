"""本模块负责生成项目中的静态分析图，包括 Pareto 图、超体积曲线、约束可行性和鲁棒性对比等。它优先保证 matplotlib 路径稳定可用。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_parent(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    return path_obj


def plot_pareto_2d(df: pd.DataFrame, path: str | Path, x_col: str = "ohmic_loss_mean", y_col: str = "Kc_mean") -> Path:
    path_obj = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="is_feasible", ax=ax)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_pareto_3d(
    df: pd.DataFrame,
    path: str | Path,
    x_col: str = "vp_std",
    y_col: str = "ohmic_loss_mean",
    z_col: str = "Kc_mean",
    color_col: str = "S11_max",
) -> Path:
    path_obj = _ensure_parent(path)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[x_col], df[y_col], df[z_col], c=df[color_col], cmap="viridis")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_hypervolume_history(df: pd.DataFrame, path: str | Path) -> Path:
    path_obj = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df, x="step", y="hypervolume", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_constraint_feasibility(df: pd.DataFrame, path: str | Path) -> Path:
    path_obj = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x_col = "step" if "step" in df.columns else "iteration"
    sns.lineplot(data=df, x=x_col, y="feasible_rate", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_ard_sensitivity(df: pd.DataFrame, path: str | Path) -> Path:
    path_obj = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="sensitivity", y="parameter", hue="output", ax=ax)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_nominal_vs_robust(df: pd.DataFrame, path: str | Path) -> Path:
    path_obj = _ensure_parent(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=df, x="nominal_neg_Kc", y="robust_neg_Kc", ax=axes[0])
    sns.scatterplot(data=df, x="nominal_feasible_rate", y="robust_feasible_rate", ax=axes[1])
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj


def plot_multifidelity_calibration(df: pd.DataFrame, path: str | Path, x_col: str = "measured", y_col: str = "corrected") -> Path:
    path_obj = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    low = min(df[x_col].min(), df[y_col].min())
    high = max(df[x_col].max(), df[y_col].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)
    return path_obj
