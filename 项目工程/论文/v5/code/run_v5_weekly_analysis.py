#!/usr/bin/env python3
"""Run the V5 weekly-granularity robustness analysis."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


HERE = Path(__file__).resolve()
V5_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V4_DIR = PROJECT_DIR / "论文" / "v4"
RESULTS = V5_DIR / "results"
FIGURES = V5_DIR / "figures"
MODELS = V5_DIR / "models"
LOGS = V5_DIR / "logs"
DATA = V5_DIR / "data_processed" / "seller_week_panel.csv"
SEED = 20260823
Q_PRIMARY = 0.20

sys.path.insert(0, str(V1_DIR / "code"))
sys.path.insert(0, str(V4_DIR / "code"))

import run_analysis as v1_analysis  # noqa: E402
import run_v4_analysis as v4_analysis  # noqa: E402
from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import (  # noqa: E402
    RobustPreprocessor,
    knockoff_diagnostics,
    repeated_knockoffs,
)
from run_deep_analysis import get_device, set_seed  # noqa: E402


def configure() -> None:
    for directory in (RESULTS, FIGURES, MODELS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "v5_weekly_analysis.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
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
    np.random.seed(SEED)
    set_seed(SEED)


def save_w(name: str, matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, columns=FEATURES)
    frame.insert(0, "repetition", np.arange(1, len(frame) + 1))
    frame.to_csv(RESULTS / f"{name}_w.csv", index=False)


def metric_row(name: str, y: np.ndarray, prediction: np.ndarray) -> dict:
    raw_y = np.expm1(y)
    raw_prediction = np.maximum(0, np.expm1(prediction))
    return {
        "model": name,
        "n_test": len(y),
        "rmse_log": float(mean_squared_error(y, prediction) ** 0.5),
        "mae_log": float(mean_absolute_error(y, prediction)),
        "r2_log": float(r2_score(y, prediction)),
        "wape_raw": float(
            np.abs(raw_y - raw_prediction).sum() / max(raw_y.sum(), 1e-12)
        ),
    }


def run_predictive_models(
    panel: pd.DataFrame, strict_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = panel.loc[panel["week"].le("2017-12-25")].copy()
    validation = panel.loc[
        panel["week"].between("2018-01-01", "2018-04-30")
    ].copy()
    test = panel.loc[panel["week"].between("2018-05-07", "2018-07-30")].copy()
    target = "log_gmv_next_week"
    preprocessor = RobustPreprocessor(FEATURES).fit(train, copula=False)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    x_test = preprocessor.transform(test)
    y_train = train[target].to_numpy(float)
    y_validation = validation[target].to_numpy(float)
    y_test = test[target].to_numpy(float)
    all_train = np.vstack([x_train, x_validation])
    all_y = np.r_[y_train, y_validation]
    rows = [
        metric_row(
            "Naive-current-GMV",
            y_test,
            test["log_gmv"].to_numpy(float),
        )
    ]

    ridge_candidates = [
        Ridge(alpha=alpha).fit(x_train, y_train)
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ]
    ridge = min(
        ridge_candidates,
        key=lambda model: mean_squared_error(
            y_validation, model.predict(x_validation)
        ),
    )
    ridge_alpha = float(ridge.alpha)
    ridge.fit(all_train, all_y)
    rows.append(metric_row("Ridge", y_test, ridge.predict(x_test)))

    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 0, 50),
        cv=5,
        max_iter=15000,
        n_jobs=-1,
        random_state=SEED,
    ).fit(x_train, y_train)
    lasso_alpha = float(lasso_cv.alpha_)
    lasso = Lasso(alpha=lasso_alpha, max_iter=15000).fit(all_train, all_y)
    rows.append(metric_row("Lasso", y_test, lasso.predict(x_test)))

    extra_candidates = [
        ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=leaf,
            max_features=features,
            n_jobs=-1,
            random_state=SEED,
        ).fit(x_train, y_train)
        for leaf, features in ((3, 0.7), (8, 1.0))
    ]
    extra = min(
        extra_candidates,
        key=lambda model: mean_squared_error(
            y_validation, model.predict(x_validation)
        ),
    )
    extra_params = {
        "min_samples_leaf": int(extra.min_samples_leaf),
        "max_features": extra.max_features,
    }
    extra.fit(all_train, all_y)
    rows.append(metric_row("ExtraTrees", y_test, extra.predict(x_test)))

    xgb_candidates = []
    for depth, rate, child in (
        (2, 0.04, 10),
        (3, 0.04, 10),
        (3, 0.08, 20),
        (4, 0.04, 20),
    ):
        model = xgb.XGBRegressor(
            n_estimators=550,
            max_depth=depth,
            learning_rate=rate,
            min_child_weight=child,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=SEED,
        ).fit(x_train, y_train)
        validation_rmse = float(
            mean_squared_error(y_validation, model.predict(x_validation))
            ** 0.5
        )
        xgb_candidates.append((validation_rmse, model))
    validation_rmse, xgb_model = min(xgb_candidates, key=lambda item: item[0])
    xgb_params = {
        "n_estimators": 550,
        "max_depth": int(xgb_model.max_depth),
        "learning_rate": float(xgb_model.learning_rate),
        "min_child_weight": float(xgb_model.min_child_weight),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": SEED,
    }
    xgb_model.fit(all_train, all_y)
    prediction_xgb = xgb_model.predict(x_test)
    rows.append(metric_row("XGBoost", y_test, prediction_xgb))

    mlp_candidates = [
        MLPRegressor(
            hidden_layer_sizes=layers,
            alpha=alpha,
            early_stopping=True,
            max_iter=500,
            random_state=SEED,
        ).fit(x_train, y_train)
        for layers, alpha in (
            ((64, 32), 0.001),
            ((64, 32), 0.01),
            ((128, 64), 0.01),
        )
    ]
    mlp = min(
        mlp_candidates,
        key=lambda model: mean_squared_error(
            y_validation, model.predict(x_validation)
        ),
    )
    mlp_params = {
        "hidden_layer_sizes": list(mlp.hidden_layer_sizes),
        "alpha": float(mlp.alpha),
    }
    mlp.fit(all_train, all_y)
    rows.append(metric_row("MLP", y_test, mlp.predict(x_test)))

    strict_features = [feature for feature in strict_features if feature in FEATURES]
    if strict_features:
        indices = [FEATURES.index(feature) for feature in strict_features]
        selected_model = clone(xgb_model).fit(all_train[:, indices], all_y)
        rows.append(
            metric_row(
                "XGBoost-weekly-FDR-set",
                y_test,
                selected_model.predict(x_test[:, indices]),
            )
        )

    dmatrix = xgb.DMatrix(x_test, feature_names=FEATURES)
    contributions = xgb_model.get_booster().predict(
        dmatrix, pred_contribs=True
    )[:, :-1]
    shap = pd.DataFrame(
        {
            "feature": FEATURES,
            "label_zh": [FEATURE_LABELS_ZH[feature] for feature in FEATURES],
            "mean_abs_shap": np.abs(contributions).mean(axis=0),
            "mean_shap": contributions.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap.to_csv(RESULTS / "weekly_xgboost_shap.csv", index=False)

    permutation = permutation_importance(
        xgb_model,
        x_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "label_zh": [FEATURE_LABELS_ZH[feature] for feature in FEATURES],
            "importance_mean": permutation.importances_mean,
            "importance_sd": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        RESULTS / "weekly_xgboost_permutation.csv", index=False
    )

    predictions = test[
        ["seller_id", "week", "gmv_next_week", "log_gmv_next_week"]
    ].copy()
    predictions["predicted_log_gmv"] = prediction_xgb
    predictions["predicted_gmv"] = np.maximum(0, np.expm1(prediction_xgb))
    predictions.to_csv(RESULTS / "weekly_test_predictions.csv", index=False)
    metrics = pd.DataFrame(rows).sort_values("rmse_log")
    metrics.to_csv(RESULTS / "weekly_predictive_metrics.csv", index=False)
    metadata = {
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "train_period": [str(train["week"].min()), str(train["week"].max())],
        "validation_period": [
            str(validation["week"].min()),
            str(validation["week"].max()),
        ],
        "test_period": [str(test["week"].min()), str(test["week"].max())],
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "extra_trees": extra_params,
        "xgboost": xgb_params,
        "xgboost_validation_rmse": validation_rmse,
        "mlp": mlp_params,
        "weekly_strict_features": strict_features,
    }
    return metrics, shap, metadata


def compare_monthly_weekly(
    weekly_primary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    monthly = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    rows = []
    summary = {}
    for q in (0.10, 0.20, 0.30):
        monthly_q = monthly.loc[monthly["q"].eq(q)]
        weekly_q = weekly_primary.loc[weekly_primary["q"].eq(q)]
        monthly_set = set(
            monthly_q.loc[monthly_q["selected_ebh"], "feature"]
        )
        weekly_set = set(weekly_q.loc[weekly_q["selected_ebh"], "feature"])
        union = monthly_set | weekly_set
        intersection = monthly_set & weekly_set
        summary[str(q)] = {
            "monthly_count": len(monthly_set),
            "weekly_count": len(weekly_set),
            "intersection_count": len(intersection),
            "jaccard": len(intersection) / len(union) if union else 1.0,
            "intersection": sorted(intersection),
            "monthly_only": sorted(monthly_set - weekly_set),
            "weekly_only": sorted(weekly_set - monthly_set),
        }
        for feature in FEATURES:
            rows.append(
                {
                    "q": q,
                    "feature": feature,
                    "label_zh": FEATURE_LABELS_ZH[feature],
                    "monthly_selected": feature in monthly_set,
                    "weekly_selected": feature in weekly_set,
                    "selected_both": feature in intersection,
                    "monthly_frequency": float(
                        monthly_q.loc[
                            monthly_q["feature"].eq(feature),
                            "selection_frequency",
                        ].iloc[0]
                    ),
                    "weekly_frequency": float(
                        weekly_q.loc[
                            weekly_q["feature"].eq(feature),
                            "selection_frequency",
                        ].iloc[0]
                    ),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "monthly_weekly_comparison.csv", index=False)
    return comparison, summary


def make_figures(
    panel: pd.DataFrame,
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    weekly = panel.groupby("week", as_index=False).agg(
        gmv=("gmv", "sum"),
        active_sellers=("seller_id", "nunique"),
        orders=("order_count", "sum"),
    )
    weekly.to_csv(RESULTS / "weekly_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly["week"], weekly["gmv"] / 1e6, color="#176B87", linewidth=1.6)
    ax.set_title("Olist周度GMV")
    ax.set_xlabel("周")
    ax.set_ylabel("GMV（百万雷亚尔）")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v5_weekly_gmv.png", dpi=220)
    plt.close(fig)

    q20 = primary.loc[primary["q"].eq(0.20)].sort_values(
        "selection_frequency"
    )
    colors = np.where(q20["selected_ebh"], "#D1495B", "#5B7083")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(q20["label_zh"], q20["selection_frequency"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("60轮单轮入选频率")
    ax.set_title("周度Copula-MVR Knockoff稳定性")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v5_weekly_selection.png", dpi=220)
    plt.close(fig)

    compare_q20 = comparison.loc[comparison["q"].eq(0.20)]
    fig, ax = plt.subplots(figsize=(7, 7))
    compare_q20 = compare_q20.copy()
    compare_q20["evidence"] = np.select(
        [
            compare_q20["selected_both"],
            compare_q20["monthly_selected"],
            compare_q20["weekly_selected"],
        ],
        ["月周共同", "仅月度", "仅周度"],
        default="均未入选",
    )
    bubble = (
        compare_q20.groupby(
            ["monthly_frequency", "weekly_frequency", "evidence"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )
    palette = {
        "月周共同": "#D1495B",
        "仅月度": "#287271",
        "仅周度": "#E9A03B",
        "均未入选": "#7A8793",
    }
    for evidence, view in bubble.groupby("evidence"):
        ax.scatter(
            view["monthly_frequency"],
            view["weekly_frequency"],
            s=45 + 28 * view["count"],
            color=palette[evidence],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label=evidence,
        )
        for _, row in view.loc[view["count"].gt(1)].iterrows():
            ax.text(
                row["monthly_frequency"],
                row["weekly_frequency"],
                str(int(row["count"])),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set_xlabel("月度入选频率")
    ax.set_ylabel("周度入选频率")
    ax.set_title("月度与周度稳定性比较")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v5_monthly_weekly_frequency.png", dpi=220)
    plt.close(fig)

    ordered = metrics.sort_values("rmse_log", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["model"], ordered["rmse_log"], color="#2A9D8F")
    ax.set_xlabel("测试集RMSE（log尺度）")
    ax.set_title("周度时间外预测")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v5_weekly_prediction.png", dpi=220)
    plt.close(fig)


def main() -> None:
    configure()
    started = time.time()
    panel = pd.read_csv(DATA, parse_dates=["week"])
    logging.info(
        "Weekly panel loaded n=%s sellers=%s weeks=%s",
        len(panel),
        panel["seller_id"].nunique(),
        panel["week"].nunique(),
    )

    primary, primary_metadata, primary_w = repeated_knockoffs(
        panel,
        "log_gmv_next_week",
        "copula",
        repetitions=60,
        seed_offset=41000,
    )
    primary.to_csv(RESULTS / "weekly_knockoff_primary.csv", index=False)
    save_w("weekly_primary_copula_lasso", primary_w)

    gaussian, gaussian_metadata, gaussian_w = repeated_knockoffs(
        panel,
        "log_gmv_next_week",
        "gaussian",
        repetitions=40,
        seed_offset=42000,
    )
    gaussian.to_csv(RESULTS / "weekly_knockoff_gaussian.csv", index=False)
    save_w("weekly_gaussian_lasso", gaussian_w)

    same_week, same_week_metadata, same_week_w = repeated_knockoffs(
        panel,
        "log_gmv",
        "copula",
        repetitions=40,
        q_values=(0.20,),
        seed_offset=43000,
    )
    same_week.to_csv(RESULTS / "weekly_same_week_sensitivity.csv", index=False)
    save_w("weekly_same_week", same_week_w)

    trimmed_panel = panel.loc[
        panel["week"].between("2017-03-06", "2018-05-28")
    ]
    trimmed, trimmed_metadata, trimmed_w = repeated_knockoffs(
        trimmed_panel,
        "log_gmv_next_week",
        "copula",
        repetitions=40,
        q_values=(0.20,),
        seed_offset=44000,
    )
    trimmed.to_csv(RESULTS / "weekly_trimmed_sensitivity.csv", index=False)
    save_w("weekly_trimmed", trimmed_w)

    aov, aov_metadata, aov_w = repeated_knockoffs(
        panel,
        "log_aov_next_week",
        "copula",
        repetitions=40,
        q_values=(0.20,),
        seed_offset=45000,
    )
    aov.to_csv(RESULTS / "weekly_aov_sensitivity.csv", index=False)
    save_w("weekly_aov", aov_w)

    pretest = panel.loc[panel["week"].le("2018-04-30")]
    pretest_result, pretest_metadata, pretest_w = repeated_knockoffs(
        pretest,
        "log_gmv_next_week",
        "copula",
        repetitions=40,
        q_values=(0.20,),
        seed_offset=46000,
    )
    pretest_result.to_csv(
        RESULTS / "weekly_pretest_knockoff.csv", index=False
    )
    save_w("weekly_pretest", pretest_w)

    ai, ai_metadata, ai_w = repeated_knockoffs(
        panel,
        "log_gmv_next_week",
        "copula",
        repetitions=25,
        q_values=(0.20,),
        stat_model="xgboost",
        seed_offset=47000,
    )
    ai.to_csv(RESULTS / "weekly_xgboost_knockoff.csv", index=False)
    save_w("weekly_xgboost_knockoff", ai_w)

    strict_features = primary.loc[
        primary["q"].eq(Q_PRIMARY) & primary["selected_ebh"], "feature"
    ].tolist()
    stable_features = primary.loc[
        primary["q"].eq(Q_PRIMARY)
        & primary["selection_frequency"].ge(0.90),
        "feature",
    ].tolist()
    metrics, shap, predictive_metadata = run_predictive_models(
        panel, strict_features
    )
    comparison, comparison_summary = compare_monthly_weekly(primary)

    # V4 routines expect monthly column names. The aliases preserve the
    # weekly values while keeping the original tested implementations intact.
    panel_v4 = panel.copy()
    panel_v4["log_gmv_next_month"] = panel_v4["log_gmv_next_week"]
    panel_v4["month"] = panel_v4["week"]
    panel_v4["year_month"] = panel_v4["year_week"]
    v4_analysis.RESULTS = RESULTS
    v4_analysis.FIGURES = FIGURES
    v4_analysis.MODELS = MODELS
    v4_analysis.LOGS = LOGS
    v1_analysis.RESULTS_DIR = RESULTS
    v1_analysis.FIGURES_DIR = FIGURES

    v4_config = v4_analysis.V4Config()
    clustered, clustered_metadata = v4_analysis.run_clustered_knockoffs(
        panel_v4, v4_config
    )
    grouped, group_metadata = v4_analysis.run_group_knockoff(
        panel_v4, v4_config
    )
    device = get_device()
    grip, grip_metadata = v4_analysis.run_grip2_style(
        panel_v4, v4_config, device
    )
    v1_analysis.fixed_effects(panel_v4)

    diagnostics = [
        knockoff_diagnostics(panel, generator)
        for generator in ("gaussian", "copula")
    ]
    (RESULTS / "weekly_knockoff_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_figures(panel, primary, comparison, metrics)

    metadata = {
        "version": "v5",
        "role": "weekly_granularity_robustness_not_replacement_of_monthly_primary",
        "seed": SEED,
        "panel_rows": len(panel),
        "panel_sellers": int(panel["seller_id"].nunique()),
        "panel_weeks": int(panel["week"].nunique()),
        "next_week_zero_rate": float(panel["gmv_next_week"].eq(0).mean()),
        "weekly_primary": primary_metadata,
        "weekly_gaussian": gaussian_metadata,
        "weekly_same_week": same_week_metadata,
        "weekly_trimmed": trimmed_metadata,
        "weekly_aov": aov_metadata,
        "weekly_pretest": pretest_metadata,
        "weekly_xgboost_knockoff": ai_metadata,
        "weekly_strict_features": strict_features,
        "weekly_stable_90pct_features": stable_features,
        "monthly_weekly_comparison": comparison_summary,
        "predictive": predictive_metadata,
        "clustered": clustered_metadata,
        "group": group_metadata,
        "grip2_style_antisymmetry": grip_metadata["antisymmetry"],
        "v4_config": asdict(v4_config),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "xgboost": xgb.__version__,
        "torch": torch.__version__,
        "elapsed_seconds": time.time() - started,
    }
    (RESULTS / "v5_weekly_reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logging.info("Weekly strict features: %s", strict_features)
    logging.info("Weekly stable features: %s", stable_features)
    logging.info("V5 weekly analysis completed in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
