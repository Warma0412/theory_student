"""Render figures exclusively from audited V7 result tables."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, FEATURES
from validate_and_summarize import LABELS, COMPARATORS

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "savefig.dpi": 180,
})
COLORS = dict(zip(COMPARATORS, plt.get_cmap("tab10").colors[:8]))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    panel = pd.read_parquet(ROOT / "data_processed/seller_month_asof.parquet")
    dictionary = pd.read_csv(ROOT / "results/feature_dictionary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    grouped = panel.groupby("target_month").agg(value=("gmv_next_month", "sum"), n=("seller_id", "size"))
    axes[0].plot(grouped.index, grouped.value / 10000, marker="o", color="#277c79")
    axes[0].set_ylabel("面板覆盖卖家次月签收确认GMV（万BRL）")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(panel.log_gmv_next_month, bins=35, color="#a34d50", alpha=.9)
    axes[1].set_xlabel("log(1 + 下一月签收确认GMV)")
    axes[1].set_ylabel("卖家月观测数")
    save(fig, "01_data")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr = panel[FEATURES].corr(method="spearman")
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(31), range(1, 32), fontsize=7)
    ax.set_yticks(range(31), dictionary.label_zh, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=.8, label="Spearman相关系数")
    save(fig, "02_correlations")

    pred = pd.read_csv(ROOT / "results/prediction_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, evaluator in zip(axes, ("xgboost", "tabm")):
        view = pred.loc[(pred.evaluator == evaluator) & (pred.k == 12)].set_index("method").reindex(COMPARATORS)
        ax.barh(range(8), view.rmse_log, color=[COLORS[m] for m in COMPARATORS])
        ax.set_yticks(range(8), [LABELS[m] for m in COMPARATORS])
        baseline = pred.loc[(pred.method == "all_features") & (pred.evaluator == evaluator), "rmse_log"].iloc[0]
        ax.axvline(baseline, color="black", linestyle="--", linewidth=1, label="31项基线")
        ax.set_title(f"相同Top-12 / {evaluator}评价器")
        ax.set_xlabel("测试RMSE（对数GMV尺度）")
        ax.set_xlim(max(0, view.rmse_log.min() - .12), view.rmse_log.max() + .1)
        for i, value in enumerate(view.rmse_log):
            ax.text(value + .002, i, f"{value:.3f}", va="center", fontsize=8)
        ax.legend(fontsize=8)
    save(fig, "03_prediction")

    stable = pd.read_csv(ROOT / "results/stability_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, fraction in zip(axes, (.5, .7, .8)):
        view = stable.loc[(stable.fraction == fraction) & (stable.k == 12)].set_index("method").reindex(COMPARATORS)
        mean = view["mean"]
        ax.errorbar(mean, range(8), xerr=np.vstack([mean - view.ci_low.clip(0, 1),
                                                   view.ci_high.clip(0, 1) - mean]),
                    fmt="o", capsize=3, color="#277c79")
        ax.set_yticks(range(8), [LABELS[m] for m in COMPARATORS], fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{fraction:.0%}卖家样本 / Top-12")
        ax.set_xlabel("与完整开发样本名单的Jaccard")
        ax.grid(axis="x", alpha=.2)
    save(fig, "04_stability")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, evaluator in zip(axes, ("xgboost", "tabm")):
        for method in COMPARATORS:
            v = pred.loc[(pred.method == method) & (pred.evaluator == evaluator)].sort_values("k")
            ax.plot(v.k, v.rmse_log, marker="o", markersize=3, label=LABELS[method], color=COLORS[method])
        ax.set_xticks([4, 8, 10, 12, 14])
        ax.set_title(evaluator)
        ax.set_xlabel("保留特征数K")
        ax.set_ylabel("测试RMSE")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "05_budget_path")

    sim = pd.read_csv(ROOT / "results/simulation_summary.csv")
    scenarios = list(json.loads((ROOT / "protocol.json").read_text())["simulation"]["scenarios"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, scenario in zip(axes.flat, scenarios):
        v = sim.loc[(sim.scenario == scenario["name"]) & (sim.rule == "top12")].set_index("method").reindex(COMPARATORS)
        ax.barh(range(8), v.power_mean, color=[COLORS[m] for m in COMPARATORS], height=.65)
        ax.set_yticks(range(8), [LABELS[m] for m in COMPARATORS], fontsize=7)
        for i, (_, row) in enumerate(v.iterrows()):
            ax.text(1.03, i, f"FDP {row.FDP_mean:.3f}", va="center", fontsize=7)
        ax.set_title(scenario["name"], fontsize=10)
        ax.set_xlabel("Power@12（同一K下FDP与Power一一对应）", fontsize=8)
        ax.set_xlim(0, 1.43)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(axis="x", alpha=.2)
    save(fig, "06_simulation_fixed_budget")

    native = sim.loc[sim.rule.eq("native")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, v in native.groupby("method"):
        v = v.set_index("scenario").reindex([s["name"] for s in scenarios])
        axes[0].plot(range(6), v.FDP_mean, marker="o", label=LABELS[method], color=COLORS[method])
        axes[1].plot(range(6), v.power_mean, marker="o", label=LABELS[method], color=COLORS[method])
    for ax in axes:
        ax.set_xticks(range(6), [s["name"].replace("_", "\n") for s in scenarios], fontsize=7)
        ax.set_ylim(-.03, 1.05)
        ax.grid(alpha=.2)
    axes[0].axhline(.2, color="black", ls="--", lw=1)
    axes[0].set_ylabel("原生名单平均FDP")
    axes[1].set_ylabel("原生名单Power")
    axes[1].legend(fontsize=8, loc="lower right")
    save(fig, "07_simulation_native")

    runs = pd.read_csv(ROOT / "results/stability_runs.csv")
    freq = []
    for method in COMPARATORS:
        z = np.zeros(31)
        records = runs.loc[(runs.method == method) & (runs.k == 12) & (runs.fraction == .5)]
        for features in records.features:
            z[json.loads(features)] += 1
        freq.append(z / len(records))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(np.array(freq).T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8), [LABELS[m] for m in COMPARATORS], rotation=35, ha="right")
    ax.set_yticks(range(31), dictionary.label_zh, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=.8, label="60个半样本中的Top-12出现频率")
    save(fig, "08_feature_frequency")

    c = json.loads((ROOT / "results/copula_diagnostic.json").read_text())
    d = json.loads((ROOT / "results/runs/deepdrk/reference.json").read_text())["metadata"]["diagnostics"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, records in [("Copula-MVR", c["classifier_tests"])] + [
        (f"DeepDRK seed {seed}", r["classifier_tests"]) for seed, r in zip(SEEDS, d)
    ]:
        ax.plot([r["swap_fraction"] for r in records], [r["auc"] for r in records], marker="o", label=name)
    ax.axhline(.5, color="black", ls="--", lw=1)
    ax.set_ylim(.45, 1.02)
    ax.set_xlabel("交换特征比例")
    ax.set_ylabel("联合交换性诊断分类器AUC")
    ax.legend(fontsize=8)
    save(fig, "09_generator_diagnostic")


if __name__ == "__main__":
    from common import SEEDS
    main()
