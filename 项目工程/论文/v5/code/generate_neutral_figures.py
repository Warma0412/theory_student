#!/usr/bin/env python3
"""Regenerate thesis figures with method-based, version-neutral labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve()
V5_DIR = HERE.parents[1]
PAPER_DIR = V5_DIR.parent
FIGURES = V5_DIR / "figures"


def configure() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Arial Unicode MS",
                "PingFang SC",
                "Heiti SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def predictive_figure() -> None:
    frame = pd.read_csv(
        PAPER_DIR / "v2" / "results" / "v2_predictive_model_metrics.csv"
    )
    names = {
        "XGBoost": "XGBoost",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ExtraTrees": "Extra Trees",
        "XGBoost-Knockoff-selected": "XGBoost（筛选变量）",
        "MLP": "MLP",
        "Naive-current-GMV": "本月GMV基线",
        "PyTorch-Residual-MLP": "残差MLP",
    }
    frame["display"] = frame["model"].map(names)
    frame = frame.sort_values("rmse_log", ascending=False)
    colors = [
        "#C74634" if model == "PyTorch-Residual-MLP" else "#2A7FB8"
        for model in frame["model"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(frame["display"], frame["rmse_log"], color=colors)
    ax.set_xlabel("测试集RMSE（log尺度）")
    ax.set_title("月度时间外预测模型比较")
    ax.set_xlim(0, max(frame["rmse_log"]) * 1.08)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_predictive_models.png", dpi=220)
    plt.close(fig)


def cluster_group_figure() -> None:
    clustered = pd.read_csv(
        PAPER_DIR / "v4" / "results" / "clustered_knockoff_selection.csv"
    )
    grouped = pd.read_csv(
        PAPER_DIR / "v4" / "results" / "group_knockoff_selection.csv"
    )
    between = (
        clustered.loc[
            clustered["method"].eq("clustered_between_seller")
            & clustered["q"].eq(0.20)
        ]
        .nlargest(14, "selection_frequency")
        .sort_values("selection_frequency")
    )
    group_view = grouped.loc[grouped["q"].eq(0.30)].sort_values(
        "selection_frequency"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    axes[0].barh(
        between["label_zh"],
        between["selection_frequency"],
        color="#2A7FB8",
    )
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title("卖家间层变量稳定频率")
    axes[0].set_xlabel("单轮入选频率")
    axes[1].barh(
        group_view["label_zh"],
        group_view["selection_frequency"],
        color="#287271",
    )
    axes[1].set_xlim(0, 1.0)
    axes[1].set_title("业务组稳定频率（q=0.30）")
    axes[1].set_xlabel("单轮入选频率")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_cluster_group.png", dpi=220)
    plt.close(fig)


def exchangeability_figure() -> None:
    frame = pd.read_csv(
        PAPER_DIR / "v4" / "results" / "v4_exchangeability_diagnostics.csv"
    )
    labels = {
        "copula_mvr": "Copula-MVR",
        "v2_deep_train_calibrated": "基础深度生成器",
        "v4_worst_swap_train_calibrated": "最坏swap深度生成器",
    }
    palette = {
        "copula_mvr": "#2A7FB8",
        "v2_deep_train_calibrated": "#E68A2E",
        "v4_worst_swap_train_calibrated": "#2A9D55",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for method, view in frame.groupby("method"):
        view = view.sort_values("swap_ratio")
        axes[0].plot(
            view["swap_ratio"],
            view["deep_kernel_mmd"],
            marker="o",
            linewidth=2,
            color=palette[method],
            label=labels[method],
        )
        axes[1].plot(
            view["swap_ratio"],
            view["classifier_auc"],
            marker="o",
            linewidth=2,
            color=palette[method],
            label=labels[method],
        )
    axes[0].set_title("深度核MMD")
    axes[0].set_xlabel("交换比例")
    axes[0].legend(frameon=False)
    axes[1].set_title("交换分类器AUC")
    axes[1].set_xlabel("交换比例")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_exchangeability.png", dpi=220)
    plt.close(fig)


def trajectory_figure() -> None:
    frame = pd.read_csv(
        PAPER_DIR / "v4" / "results" / "grip2_style_selection.csv"
    )
    view = (
        frame.loc[frame["q"].eq(0.20)]
        .nlargest(15, "mean_w")
        .sort_values("mean_w")
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(view["label_zh"], view["mean_w"], color="#C74634")
    ax.set_xlabel("平均W")
    ax.set_title("轨迹聚合非线性重要性")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_grip_importance.png", dpi=220)
    plt.close(fig)


def weekly_predictive_figure() -> None:
    frame = pd.read_csv(V5_DIR / "results" / "weekly_predictive_metrics.csv")
    names = {
        "ExtraTrees": "Extra Trees",
        "XGBoost": "XGBoost",
        "XGBoost-weekly-FDR-set": "XGBoost（周度18项）",
        "Lasso": "Lasso",
        "Ridge": "Ridge",
        "MLP": "MLP",
        "Naive-current-GMV": "本周GMV基线",
    }
    frame["display"] = frame["model"].map(names)
    frame = frame.sort_values("rmse_log", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(frame["display"], frame["rmse_log"], color="#2A9D8F")
    ax.set_xlabel("测试集RMSE（log尺度）")
    ax.set_title("周度时间外预测")
    ax.set_xlim(0, max(frame["rmse_log"]) * 1.08)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig18_weekly_prediction.png", dpi=220)
    plt.close(fig)


def main() -> None:
    configure()
    predictive_figure()
    cluster_group_figure()
    exchangeability_figure()
    trajectory_figure()
    weekly_predictive_figure()
    print("Generated five version-neutral figures.")


if __name__ == "__main__":
    main()
