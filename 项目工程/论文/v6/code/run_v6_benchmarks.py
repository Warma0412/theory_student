#!/usr/bin/env python3
"""Run V6 benchmark comparisons and component ablations on the monthly panel."""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from knockpy.knockoffs import GaussianSampler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve()
V6_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V5_DIR = PROJECT_DIR / "论文" / "v5"
RESULTS = V6_DIR / "results"
FIGURES = V6_DIR / "figures"
LOGS = V6_DIR / "logs"
DATA = V6_DIR / "data_processed" / "seller_month_panel.csv"
SEED = 20260907
Q = 0.20

sys.path.insert(0, str(V1_DIR / "code"))
from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import (  # noqa: E402
    RobustPreprocessor,
    choose_lasso_alpha,
    ebh,
    lasso_w,
    threshold_knockoff_plus,
)


def configure() -> None:
    for directory in (RESULTS, FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "v6_benchmarks.log", mode="w"),
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


def metric_row(method: str, y: np.ndarray, prediction: np.ndarray) -> dict:
    raw_y = np.expm1(y)
    raw_pred = np.maximum(0, np.expm1(prediction))
    return {
        "method": method,
        "n_test": len(y),
        "rmse_log": float(mean_squared_error(y, prediction) ** 0.5),
        "mae_log": float(mean_absolute_error(y, prediction)),
        "r2_log": float(r2_score(y, prediction)),
        "wape_raw": float(
            np.abs(raw_y - raw_pred).sum() / max(raw_y.sum(), 1e-12)
        ),
    }


def jaccard(left: set[int] | set[str], right: set[int] | set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def aggregate_w(
    w_matrix: np.ndarray, q: float = Q
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = w_matrix.shape[1]
    e_runs = np.zeros_like(w_matrix)
    selected_runs = np.zeros_like(w_matrix, dtype=bool)
    for row, w in enumerate(w_matrix):
        threshold = threshold_knockoff_plus(w, q / 2)
        if np.isfinite(threshold):
            selected = w >= threshold
            denominator = 1 + np.sum(w <= -threshold)
            e_runs[row, selected] = p / denominator
            selected_runs[row] = selected
    mean_e = e_runs.mean(axis=0)
    selected = ebh(mean_e, q)
    return selected, selected_runs.mean(axis=0), mean_e


def choose_regularization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[float, float, float]:
    candidates = []
    for l1_ratio in (0.25, 0.50, 0.75, 0.90, 1.0):
        for alpha in np.logspace(-4, -0.2, 32):
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=l1_ratio,
                max_iter=15000,
                selection="cyclic",
            ).fit(x_train, y_train)
            rmse = mean_squared_error(y_valid, model.predict(x_valid)) ** 0.5
            candidates.append((rmse, float(alpha), float(l1_ratio)))
    _, elastic_alpha, elastic_l1 = min(candidates)
    lasso_candidates = []
    for alpha in np.logspace(-4, -0.2, 45):
        model = Lasso(alpha=float(alpha), max_iter=15000).fit(x_train, y_train)
        rmse = mean_squared_error(y_valid, model.predict(x_valid)) ** 0.5
        lasso_candidates.append((rmse, float(alpha)))
    _, lasso_alpha = min(lasso_candidates)
    return elastic_alpha, elastic_l1, lasso_alpha


def seller_subsample_indices(
    sellers: np.ndarray,
    rng: np.random.Generator,
    fraction: float = 0.70,
) -> np.ndarray:
    unique = np.unique(sellers)
    chosen = rng.choice(
        unique, size=max(2, int(len(unique) * fraction)), replace=False
    )
    return np.flatnonzero(np.isin(sellers, chosen))


def stability_selection(
    x: np.ndarray,
    y: np.ndarray,
    sellers: np.ndarray,
    alpha: float,
    repetitions: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = []
    for rep in range(repetitions):
        rng = np.random.default_rng(SEED + 1000 + rep)
        indices = seller_subsample_indices(sellers, rng)
        model = Lasso(alpha=alpha, max_iter=15000).fit(x[indices], y[indices])
        coefficients.append(np.abs(model.coef_))
    matrix = np.vstack(coefficients)
    frequency = (matrix > 1e-10).mean(axis=0)
    score = frequency + matrix.mean(axis=0) * 1e-4
    return score, matrix


def shadow_extra_trees(
    x: np.ndarray,
    y: np.ndarray,
    sellers: np.ndarray,
    repetitions: int = 40,
    trees: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    wins = []
    p = x.shape[1]
    for rep in range(repetitions):
        rng = np.random.default_rng(SEED + 3000 + rep)
        indices = seller_subsample_indices(sellers, rng)
        observed = x[indices]
        shadow = observed.copy()
        for column in range(p):
            shadow[:, column] = rng.permutation(shadow[:, column])
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=8,
            max_features=0.7,
            n_jobs=-1,
            random_state=SEED + rep,
        ).fit(np.column_stack([observed, shadow]), y[indices])
        real_importance = model.feature_importances_[:p]
        shadow_max = model.feature_importances_[p:].max()
        scores.append(real_importance)
        wins.append(real_importance > shadow_max)
    score_matrix = np.vstack(scores)
    win_matrix = np.vstack(wins)
    score = win_matrix.mean(axis=0) + score_matrix.mean(axis=0) * 1e-3
    return score, score_matrix


def xgboost_shap_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    candidates = []
    for depth, rate, child in (
        (2, 0.04, 10),
        (3, 0.04, 10),
        (3, 0.08, 20),
        (4, 0.04, 20),
    ):
        params = {
            "n_estimators": 450,
            "max_depth": depth,
            "learning_rate": rate,
            "min_child_weight": child,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": SEED,
        }
        model = xgb.XGBRegressor(**params).fit(x_train, y_train)
        rmse = mean_squared_error(y_valid, model.predict(x_valid)) ** 0.5
        candidates.append((rmse, params, model))
    _, params, model = min(candidates, key=lambda item: item[0])
    contributions = model.get_booster().predict(
        xgb.DMatrix(x_valid, feature_names=FEATURES),
        pred_contribs=True,
    )[:, :-1]
    return np.abs(contributions).mean(axis=0), params


def bootstrap_xgboost_stability(
    x: np.ndarray,
    y: np.ndarray,
    sellers: np.ndarray,
    params: dict,
    reference: set[int],
    budget: int,
    repetitions: int = 15,
) -> float:
    values = []
    for rep in range(repetitions):
        rng = np.random.default_rng(SEED + 5000 + rep)
        indices = seller_subsample_indices(sellers, rng)
        local_params = params | {
            "n_estimators": 220,
            "random_state": SEED + rep,
        }
        model = xgb.XGBRegressor(**local_params).fit(x[indices], y[indices])
        contribution = model.get_booster().predict(
            xgb.DMatrix(x[indices], feature_names=FEATURES),
            pred_contribs=True,
        )[:, :-1]
        score = np.abs(contribution).mean(axis=0)
        selected = set(np.argsort(-score)[:budget])
        values.append(jaccard(selected, reference))
    return float(np.mean(values))


def run_real_data_benchmarks(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = panel.loc[panel["month"].le("2017-12-01")].copy()
    valid = panel.loc[
        panel["month"].between("2018-01-01", "2018-04-01")
    ].copy()
    test = panel.loc[
        panel["month"].between("2018-05-01", "2018-07-01")
    ].copy()
    selection = panel.loc[panel["month"].le("2018-04-01")].copy()
    target = "log_gmv_next_month"

    prep = RobustPreprocessor(FEATURES).fit(train, copula=False)
    x_train = prep.transform(train)
    x_valid = prep.transform(valid)
    x_test = prep.transform(test)
    x_selection = prep.transform(selection)
    y_train = train[target].to_numpy(float)
    y_valid = valid[target].to_numpy(float)
    y_test = test[target].to_numpy(float)
    y_selection = selection[target].to_numpy(float)
    sellers = selection["seller_id"].to_numpy()
    x_fit = np.vstack([x_train, x_valid])
    y_fit = np.r_[y_train, y_valid]

    primary = pd.read_csv(V1_DIR / "results" / "knockoff_all_scenarios.csv")
    knockoff = primary.loc[
        primary["generator"].eq("copula_pretest")
        & primary["q"].eq(Q)
    ].copy()
    knockoff_indices = [
        FEATURES.index(feature)
        for feature in knockoff.loc[knockoff["selected_ebh"], "feature"]
    ]
    budget = len(knockoff_indices)
    if budget == 0:
        raise RuntimeError("Pretest knockoff set is empty")

    start = time.time()
    elastic_alpha, elastic_l1, lasso_alpha = choose_regularization(
        x_train, y_train, x_valid, y_valid
    )
    elastic = ElasticNet(
        alpha=elastic_alpha,
        l1_ratio=elastic_l1,
        max_iter=15000,
    ).fit(x_selection, y_selection)
    elastic_score = np.abs(elastic.coef_)
    elastic_runtime = time.time() - start

    start = time.time()
    stability_score, stability_matrix = stability_selection(
        x_selection, y_selection, sellers, lasso_alpha
    )
    stability_runtime = time.time() - start

    start = time.time()
    shadow_score, shadow_matrix = shadow_extra_trees(
        x_selection, y_selection, sellers
    )
    shadow_runtime = time.time() - start

    start = time.time()
    xgb_score, xgb_params = xgboost_shap_scores(
        x_train, y_train, x_valid, y_valid
    )
    xgb_runtime = time.time() - start

    knockoff_score = (
        knockoff.set_index("feature")
        .reindex(FEATURES)["mean_evalue"]
        .to_numpy(float)
    )
    rankings = {
        "去随机化Knockoff": knockoff_score,
        "Elastic Net": elastic_score,
        "稳定性选择": stability_score,
        "影子变量树模型": shadow_score,
        "XGBoost-SHAP": xgb_score,
    }
    native = {
        "去随机化Knockoff": set(knockoff_indices),
        "Elastic Net": set(np.flatnonzero(elastic_score > 1e-10)),
        "稳定性选择": set(np.flatnonzero(stability_score >= 0.90)),
        "影子变量树模型": set(np.flatnonzero(shadow_score >= 0.80)),
        "XGBoost-SHAP": set(np.argsort(-xgb_score)[:budget]),
    }
    runtimes = {
        "去随机化Knockoff": np.nan,
        "Elastic Net": elastic_runtime,
        "稳定性选择": stability_runtime,
        "影子变量树模型": shadow_runtime,
        "XGBoost-SHAP": xgb_runtime,
    }
    reference = set(knockoff_indices)
    matched = {
        method: set(np.argsort(-score)[:budget])
        for method, score in rankings.items()
    }

    pretest_w = pd.read_csv(
        V1_DIR / "results" / "knockoff_runs" / "pretest_copula_lasso_w.csv"
    ).drop(columns="repetition")
    knock_stability = np.mean(
        [
            jaccard(
                set(np.argsort(-row)[:budget]),
                reference,
            )
            for row in pretest_w.to_numpy()
        ]
    )
    stability_values = {
        "去随机化Knockoff": float(knock_stability),
        "Elastic Net": np.nan,
        "稳定性选择": float(
            np.mean(
                [
                    jaccard(
                        set(np.argsort(-row)[:budget]),
                        matched["稳定性选择"],
                    )
                    for row in stability_matrix
                ]
            )
        ),
        "影子变量树模型": float(
            np.mean(
                [
                    jaccard(
                        set(np.argsort(-row)[:budget]),
                        matched["影子变量树模型"],
                    )
                    for row in shadow_matrix
                ]
            )
        ),
        "XGBoost-SHAP": bootstrap_xgboost_stability(
            x_selection,
            y_selection,
            sellers,
            xgb_params,
            matched["XGBoost-SHAP"],
            budget,
        ),
    }
    # Elastic Net resampling stability is computed with its fixed validation-tuned
    # regularization to keep the comparison aligned with the other baselines.
    elastic_sets = []
    for rep in range(30):
        rng = np.random.default_rng(SEED + 7000 + rep)
        indices = seller_subsample_indices(sellers, rng)
        model = ElasticNet(
            alpha=elastic_alpha,
            l1_ratio=elastic_l1,
            max_iter=15000,
        ).fit(x_selection[indices], y_selection[indices])
        elastic_sets.append(set(np.argsort(-np.abs(model.coef_))[:budget]))
    stability_values["Elastic Net"] = float(
        np.mean(
            [
                jaccard(value, matched["Elastic Net"])
                for value in elastic_sets
            ]
        )
    )

    summary_rows = []
    detail_rows = []
    prediction_rows = [
        metric_row("全部31维", y_test, fit_xgb(x_fit, y_fit, x_test, xgb_params))
    ]
    for method, score in rankings.items():
        selected = matched[method]
        indices = sorted(selected)
        prediction = fit_xgb(
            x_fit[:, indices],
            y_fit,
            x_test[:, indices],
            xgb_params,
        )
        prediction_rows.append(metric_row(method, y_test, prediction))
        summary_rows.append(
            {
                "method": method,
                "native_count": len(native[method]),
                "matched_budget": budget,
                "overlap_with_knockoff": len(selected & reference),
                "jaccard_with_knockoff": jaccard(selected, reference),
                "resampling_stability": stability_values[method],
                "selection_runtime_seconds": runtimes[method],
            }
        )
        ranks = np.empty(len(FEATURES), dtype=int)
        order = np.argsort(-score)
        ranks[order] = np.arange(1, len(FEATURES) + 1)
        for index, feature in enumerate(FEATURES):
            detail_rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "label_zh": FEATURE_LABELS_ZH[feature],
                    "score": float(score[index]),
                    "rank": int(ranks[index]),
                    "selected_native": index in native[method],
                    "selected_matched": index in selected,
                }
            )
    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)
    details = pd.DataFrame(detail_rows)
    summary.to_csv(RESULTS / "benchmark_selection_summary.csv", index=False)
    predictions.to_csv(
        RESULTS / "benchmark_prediction_metrics.csv", index=False
    )
    details.to_csv(RESULTS / "benchmark_feature_rankings.csv", index=False)
    metadata = {
        "selection_cutoff": "2018-04-01",
        "test_period": ["2018-05-01", "2018-07-01"],
        "matched_budget": budget,
        "knockoff_reference_features": [FEATURES[i] for i in sorted(reference)],
        "elastic_net": {
            "alpha": elastic_alpha,
            "l1_ratio": elastic_l1,
        },
        "stability_lasso_alpha": lasso_alpha,
        "xgboost": xgb_params,
    }
    return summary.merge(predictions, on="method", how="outer"), metadata


def fit_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict,
) -> np.ndarray:
    return xgb.XGBRegressor(**params).fit(x_train, y_train).predict(x_test)


def run_smatrix_ablation(panel: pd.DataFrame) -> pd.DataFrame:
    prep = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    x = prep.transform(panel)
    y = panel["log_gmv_next_month"].to_numpy(float)
    y = (y - y.mean()) / y.std(ddof=0)
    sigma = LedoitWolf().fit(x).covariance_
    existing_w = pd.read_csv(
        V1_DIR / "results" / "knockoff_runs" / "primary_copula_lasso_w.csv"
    ).drop(columns="repetition").to_numpy()
    alpha = float(
        json.loads(
            (V1_DIR / "results" / "knockoff_metadata.json").read_text()
        )[0]["lasso_alpha"]
    )
    reference, _, _ = aggregate_w(existing_w, Q)
    reference_set = set(reference)
    rows = []
    matrices = {"mvr": existing_w[:30]}
    for method in ("sdp", "equicorrelated"):
        started = time.time()
        sampler = GaussianSampler(
            X=x,
            mu=x.mean(axis=0),
            Sigma=sigma,
            method=method,
        )
        values = []
        for rep in range(30):
            np.random.seed(SEED + 10000 + rep)
            xk = sampler.sample_knockoffs(check_psd=True)
            values.append(lasso_w(x, xk, y, alpha))
        matrices[method] = np.vstack(values)
        pd.DataFrame(matrices[method], columns=FEATURES).to_csv(
            RESULTS / f"ablation_{method}_w.csv", index=False
        )
        logging.info("%s S-matrix ablation completed %.1fs", method, time.time() - started)

    for method, matrix in matrices.items():
        selected, frequency, mean_e = aggregate_w(matrix, Q)
        selected_set = set(selected)
        rows.append(
            {
                "ablation": "S矩阵",
                "setting": method,
                "repetitions": len(matrix),
                "selected_count": len(selected),
                "overlap_with_main": len(selected_set & reference_set),
                "jaccard_with_main": jaccard(selected_set, reference_set),
                "median_frequency_selected": float(
                    np.median(frequency[selected]) if len(selected) else 0
                ),
                "max_mean_evalue": float(mean_e.max()),
            }
        )

    # Prefix ablation reuses the exact 60 primary W runs.
    for repetitions in (1, 5, 10, 20, 40, 60):
        selected, frequency, mean_e = aggregate_w(
            existing_w[:repetitions], Q
        )
        selected_set = set(selected)
        rows.append(
            {
                "ablation": "重复次数",
                "setting": f"M={repetitions}",
                "repetitions": repetitions,
                "selected_count": len(selected),
                "overlap_with_main": len(selected_set & reference_set),
                "jaccard_with_main": jaccard(selected_set, reference_set),
                "median_frequency_selected": float(
                    np.median(frequency[selected]) if len(selected) else 0
                ),
                "max_mean_evalue": float(mean_e.max()),
            }
        )

    scenarios = {
        "Copula-MVR-Lasso": (
            V1_DIR / "results" / "knockoff_runs" / "primary_copula_lasso_w.csv"
        ),
        "原始高斯-MVR-Lasso": (
            V1_DIR / "results" / "knockoff_runs" / "gaussian_lasso_w.csv"
        ),
        "Copula-MVR-XGBoost": (
            V1_DIR / "results" / "knockoff_runs" / "copula_xgboost_w.csv"
        ),
    }
    for label, path in scenarios.items():
        matrix = pd.read_csv(path).drop(columns="repetition").to_numpy()
        selected, frequency, mean_e = aggregate_w(matrix, Q)
        selected_set = set(selected)
        rows.append(
            {
                "ablation": "边际与统计量",
                "setting": label,
                "repetitions": len(matrix),
                "selected_count": len(selected),
                "overlap_with_main": len(selected_set & reference_set),
                "jaccard_with_main": jaccard(selected_set, reference_set),
                "median_frequency_selected": float(
                    np.median(frequency[selected]) if len(selected) else 0
                ),
                "max_mean_evalue": float(mean_e.max()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "component_ablation.csv", index=False)
    return frame


def simulated_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    prep = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    full_x = prep.transform(panel)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(full_x), size=4000, replace=False)
    x = full_x[sample]
    sigma = LedoitWolf().fit(x).covariance_
    sampler = GaussianSampler(
        X=x,
        mu=x.mean(axis=0),
        Sigma=sigma,
        method="mvr",
    )
    true_features = [
        "avg_price",
        "avg_freight",
        "avg_product_name_length",
        "avg_review_score",
        "avg_ship_days",
        "item_count",
        "order_count",
        "category_count",
        "unique_customer_count",
        "buyer_state_diversity",
    ]
    true_idx = np.array([FEATURES.index(feature) for feature in true_features])
    beta = np.zeros(len(FEATURES))
    beta[true_idx] = np.array(
        [0.65, 0.45, 0.35, -0.40, -0.45, 0.55, 0.70, 0.40, 0.75, 0.35]
    )
    signal = x @ beta
    noise_sd = signal.std() / 1.8
    methods = [
        "去随机化Knockoff",
        "单次Knockoff",
        "Elastic Net",
        "稳定性选择",
        "影子变量树模型",
    ]
    records = {method: [] for method in methods}
    fixed_lasso_alpha = None
    fixed_elastic = None
    simulation_repetitions = 20
    for simulation in range(simulation_repetitions):
        local_rng = np.random.default_rng(SEED + 20000 + simulation)
        y = signal + local_rng.normal(0, noise_sd, len(x))
        y = (y - y.mean()) / y.std(ddof=0)
        w_runs = []
        for rep in range(10):
            np.random.seed(SEED + 30000 + simulation * 10 + rep)
            xk = sampler.sample_knockoffs(check_psd=True)
            if fixed_lasso_alpha is None:
                fixed_lasso_alpha = choose_lasso_alpha(x, xk, y)
            w_runs.append(lasso_w(x, xk, y, fixed_lasso_alpha))
        w_matrix = np.vstack(w_runs)
        derand, _, _ = aggregate_w(w_matrix, Q)
        single_threshold = threshold_knockoff_plus(w_matrix[0], Q)
        single = (
            np.flatnonzero(w_matrix[0] >= single_threshold)
            if np.isfinite(single_threshold)
            else np.array([], dtype=int)
        )
        if fixed_elastic is None:
            candidates = []
            train_idx = np.arange(0, 3000)
            valid_idx = np.arange(3000, 4000)
            for l1_ratio in (0.50, 0.75, 0.90, 1.0):
                for alpha in np.logspace(-3.5, -0.3, 24):
                    model = ElasticNet(
                        alpha=float(alpha),
                        l1_ratio=l1_ratio,
                        max_iter=12000,
                    ).fit(x[train_idx], y[train_idx])
                    rmse = mean_squared_error(
                        y[valid_idx], model.predict(x[valid_idx])
                    )
                    candidates.append((rmse, float(alpha), float(l1_ratio)))
            _, alpha, ratio = min(candidates)
            fixed_elastic = (alpha, ratio)
        elastic = ElasticNet(
            alpha=fixed_elastic[0],
            l1_ratio=fixed_elastic[1],
            max_iter=12000,
        ).fit(x, y)
        elastic_selected = np.flatnonzero(np.abs(elastic.coef_) > 1e-10)

        coef_runs = []
        for rep in range(20):
            sub_rng = np.random.default_rng(
                SEED + 40000 + simulation * 20 + rep
            )
            indices = sub_rng.choice(len(x), size=2800, replace=False)
            model = Lasso(alpha=fixed_lasso_alpha, max_iter=12000).fit(
                x[indices], y[indices]
            )
            coef_runs.append(np.abs(model.coef_))
        stability_frequency = (np.vstack(coef_runs) > 1e-10).mean(axis=0)
        stability_selected = np.flatnonzero(stability_frequency >= 0.90)

        shadow_wins = []
        for rep in range(8):
            shadow_rng = np.random.default_rng(
                SEED + 50000 + simulation * 8 + rep
            )
            shadow = x.copy()
            for column in range(x.shape[1]):
                shadow[:, column] = shadow_rng.permutation(shadow[:, column])
            model = ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=8,
                max_features=0.7,
                n_jobs=-1,
                random_state=SEED + simulation * 8 + rep,
            ).fit(np.column_stack([x, shadow]), y)
            importance = model.feature_importances_
            shadow_wins.append(
                importance[: len(FEATURES)] > importance[len(FEATURES) :].max()
            )
        shadow_selected = np.flatnonzero(
            np.vstack(shadow_wins).mean(axis=0) >= 0.75
        )
        selections = {
            "去随机化Knockoff": derand,
            "单次Knockoff": single,
            "Elastic Net": elastic_selected,
            "稳定性选择": stability_selected,
            "影子变量树模型": shadow_selected,
        }
        true_set = set(true_idx)
        for method, selected in selections.items():
            selected_set = set(selected)
            false = len(selected_set - true_set)
            true = len(selected_set & true_set)
            records[method].append(
                {
                    "fdp": false / max(len(selected_set), 1),
                    "power": true / len(true_set),
                    "discoveries": len(selected_set),
                }
            )
        logging.info(
            "Simulation %s/%s completed",
            simulation + 1,
            simulation_repetitions,
        )
    rows = []
    for method, values in records.items():
        frame = pd.DataFrame(values)
        rows.append(
            {
                "method": method,
                "simulations": simulation_repetitions,
                "mean_fdp": frame["fdp"].mean(),
                "probability_fdp_above_q": (frame["fdp"] > Q).mean(),
                "mean_power": frame["power"].mean(),
                "mean_discoveries": frame["discoveries"].mean(),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "simulation_method_comparison.csv", index=False)
    (RESULTS / "simulation_design.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": len(x),
                "p": len(FEATURES),
                "true_features": true_features,
                "signal_to_noise_sd_ratio": 1.8,
                "simulation_repetitions": simulation_repetitions,
                "knockoff_repetitions": 10,
                "stability_subsamples": 20,
                "shadow_repetitions": 8,
                "q": Q,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def make_figures(
    benchmark: pd.DataFrame,
    ablation: pd.DataFrame,
    simulation: pd.DataFrame,
) -> None:
    view = benchmark.sort_values("rmse_log", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].barh(view["method"], view["rmse_log"], color="#2A7FB8")
    axes[0].set_xlabel("测试RMSE（log尺度）")
    axes[0].set_title("同预算变量选择的时间外预测")
    axes[1].barh(
        view["method"],
        view["resampling_stability"],
        color="#287271",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("重抽样集合Jaccard")
    axes[1].set_title("同预算选择稳定性")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig19_benchmark_comparison.png", dpi=220)
    plt.close(fig)

    prefix = ablation.loc[ablation["ablation"].eq("重复次数")].copy()
    prefix["M"] = prefix["repetitions"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        prefix["M"],
        prefix["selected_count"],
        marker="o",
        label="严格入选数",
    )
    ax.plot(
        prefix["M"],
        prefix["overlap_with_main"],
        marker="s",
        label="与60轮主集重合",
    )
    ax.set_xlabel("Knockoff重复次数M")
    ax.set_ylabel("变量数")
    ax.set_title("重复次数消融")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig20_repetition_ablation.png", dpi=220)
    plt.close(fig)

    figure = simulation.sort_values("mean_power")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].barh(figure["method"], figure["mean_fdp"], color="#D1495B")
    axes[0].axvline(Q, linestyle="--", color="#333333", label="目标q=0.20")
    axes[0].set_xlabel("平均FDP")
    axes[0].set_title("模拟误选比例")
    axes[0].legend(frameon=False)
    axes[1].barh(figure["method"], figure["mean_power"], color="#2A9D8F")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("平均Power")
    axes[1].set_title("模拟检出能力")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig21_simulation_fdp_power.png", dpi=220)
    plt.close(fig)


def main() -> None:
    configure()
    started = time.time()
    panel = pd.read_csv(DATA, parse_dates=["month"])
    benchmark, benchmark_metadata = run_real_data_benchmarks(panel)
    ablation = run_smatrix_ablation(panel)
    simulation = simulated_comparison(panel)
    make_figures(benchmark, ablation, simulation)
    metadata = {
        "seed": SEED,
        "panel_rows": len(panel),
        "panel_sellers": int(panel["seller_id"].nunique()),
        "benchmark": benchmark_metadata,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "elapsed_seconds": time.time() - started,
    }
    (RESULTS / "v6_reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("V6 benchmarks completed in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
