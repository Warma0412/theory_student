#!/usr/bin/env python3
"""Run a paired, seller-level stability benchmark for all V6 methods."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from knockpy.knockoffs import GaussianSampler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso


HERE = Path(__file__).resolve()
V6_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
RESULTS = V6_DIR / "results"
FIGURES = V6_DIR / "figures"
LOGS = V6_DIR / "logs"
DATA = V6_DIR / "data_processed" / "seller_month_panel.csv"
SEED = 20260908
Q = 0.20
BUDGET = 12
METHODS = [
    "去随机化Knockoff",
    "Elastic Net",
    "稳定性选择",
    "影子变量树模型",
    "XGBoost-SHAP",
]

import sys

sys.path.insert(0, str(V1_DIR / "code"))
from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import (  # noqa: E402
    RobustPreprocessor,
    lasso_w,
)

sys.path.insert(0, str(HERE.parent))
from run_v6_benchmarks import aggregate_w, jaccard  # noqa: E402


def configure() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "v6_fair_stability.log", mode="w"),
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


def top_set(score: np.ndarray, budget: int = BUDGET) -> set[int]:
    return set(np.argsort(-np.asarray(score), kind="stable")[:budget])


def knockoff_top_set(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    seed: int,
    repetitions: int,
) -> tuple[set[int], int]:
    y_scaled = (y - y.mean()) / y.std(ddof=0)
    sigma = LedoitWolf().fit(x).covariance_
    sampler = GaussianSampler(
        X=x,
        mu=x.mean(axis=0),
        Sigma=sigma,
        method="mvr",
    )
    runs = []
    for rep in range(repetitions):
        np.random.seed(seed + rep)
        xk = sampler.sample_knockoffs(check_psd=True)
        runs.append(lasso_w(x, xk, y_scaled, alpha))
    matrix = np.vstack(runs)
    strict, _, mean_e = aggregate_w(matrix, Q)
    mean_w = matrix.mean(axis=0)
    score = mean_e + 1e-6 * (mean_w - mean_w.min())
    return top_set(score), len(strict)


def elastic_top_set(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> tuple[set[int], int]:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=15000,
    ).fit(x, y)
    score = np.abs(model.coef_)
    return top_set(score), int(np.sum(score > 1e-10))


def stability_top_set(
    x: np.ndarray,
    y: np.ndarray,
    sellers: np.ndarray,
    alpha: float,
    seed: int,
    repetitions: int,
) -> tuple[set[int], int]:
    unique = np.unique(sellers)
    coefficients = []
    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        chosen = rng.choice(
            unique,
            size=max(2, int(len(unique) * 0.70)),
            replace=False,
        )
        indices = np.flatnonzero(np.isin(sellers, chosen))
        model = Lasso(alpha=alpha, max_iter=15000).fit(x[indices], y[indices])
        coefficients.append(np.abs(model.coef_))
    matrix = np.vstack(coefficients)
    frequency = (matrix > 1e-10).mean(axis=0)
    score = frequency + matrix.mean(axis=0) * 1e-4
    return top_set(score), int(np.sum(frequency >= 0.90))


def shadow_top_set(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    repetitions: int,
) -> tuple[set[int], int]:
    score_rows = []
    win_rows = []
    p = x.shape[1]
    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        shadow = x.copy()
        for column in range(p):
            shadow[:, column] = rng.permutation(shadow[:, column])
        model = ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=8,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed + rep,
        ).fit(np.column_stack([x, shadow]), y)
        importance = model.feature_importances_
        real = importance[:p]
        score_rows.append(real)
        win_rows.append(real > importance[p:].max())
    scores = np.vstack(score_rows)
    wins = np.vstack(win_rows)
    frequency = wins.mean(axis=0)
    score = frequency + scores.mean(axis=0) * 1e-3
    return top_set(score), int(np.sum(frequency >= 0.75))


def xgb_shap_top_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    params: dict,
    seed: int,
) -> tuple[set[int], int]:
    local = params | {
        "n_jobs": -1,
        "random_state": seed,
    }
    model = xgb.XGBRegressor(**local).fit(x_train, y_train)
    contributions = model.get_booster().predict(
        xgb.DMatrix(x_valid, feature_names=FEATURES),
        pred_contribs=True,
    )[:, :-1]
    score = np.abs(contributions).mean(axis=0)
    return top_set(score), BUDGET


def load_reference_sets() -> dict[str, set[int]]:
    ranking = pd.read_csv(RESULTS / "benchmark_feature_rankings.csv")
    references = {}
    for method in METHODS:
        selected = ranking.loc[
            ranking["method"].eq(method)
            & ranking["selected_matched"].astype(bool),
            "feature",
        ]
        references[method] = {FEATURES.index(value) for value in selected}
        if len(references[method]) != BUDGET:
            raise ValueError(f"{method} reference size is not {BUDGET}")
    return references


def load_parameters() -> tuple[float, float, float, float, dict]:
    metadata = json.loads(
        (RESULTS / "v6_reproducibility.json").read_text(encoding="utf-8")
    )["benchmark"]
    knockoff_metadata = json.loads(
        (V1_DIR / "results" / "knockoff_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    pretest = next(
        row for row in knockoff_metadata if row.get("generator") == "copula_pretest"
    )
    return (
        float(pretest["lasso_alpha"]),
        float(metadata["elastic_net"]["alpha"]),
        float(metadata["elastic_net"]["l1_ratio"]),
        float(metadata["stability_lasso_alpha"]),
        metadata["xgboost"],
    )


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in runs.groupby("method", sort=False):
        values = frame["jaccard"].to_numpy(float)
        standard_error = values.std(ddof=1) / math.sqrt(len(values))
        rows.append(
            {
                "method": method,
                "outer_repetitions": len(values),
                "seller_fraction": frame["seller_fraction"].iloc[0],
                "matched_budget": BUDGET,
                "mean_overlap": frame["overlap"].mean(),
                "mean_jaccard": values.mean(),
                "std_jaccard": values.std(ddof=1),
                "ci95_lower": max(0.0, values.mean() - 1.96 * standard_error),
                "ci95_upper": min(1.0, values.mean() + 1.96 * standard_error),
                "minimum_jaccard": values.min(),
                "maximum_jaccard": values.max(),
                "exact_match_rate": frame["exact_match"].mean(),
                "mean_native_count": frame["native_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    pivot = runs.pivot(
        index="outer_repetition",
        columns="method",
        values="jaccard",
    )
    knockoff = pivot["去随机化Knockoff"]
    rows = []
    for method in METHODS:
        if method == "去随机化Knockoff":
            continue
        difference = knockoff - pivot[method]
        standard_error = difference.std(ddof=1) / math.sqrt(len(difference))
        rows.append(
            {
                "comparison": f"去随机化Knockoff - {method}",
                "paired_repetitions": len(difference),
                "mean_difference": difference.mean(),
                "std_difference": difference.std(ddof=1),
                "ci95_lower": difference.mean() - 1.96 * standard_error,
                "ci95_upper": difference.mean() + 1.96 * standard_error,
                "knockoff_higher_rate": (difference > 0).mean(),
                "tie_rate": (difference == 0).mean(),
                "knockoff_lower_rate": (difference < 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    view = summary.sort_values("mean_jaccard")
    lower = view["mean_jaccard"] - view["ci95_lower"]
    upper = view["ci95_upper"] - view["mean_jaccard"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [
        "#7C434C" if method == "去随机化Knockoff" else "#287271"
        for method in view["method"]
    ]
    ax.barh(
        view["method"],
        view["mean_jaccard"],
        xerr=np.vstack([lower, upper]),
        color=colors,
        alpha=0.92,
        capsize=4,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("同一70%卖家子样本协议下的平均Jaccard（95% CI）")
    ax.set_title("五种方法的配对卖家子抽样稳定性")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig22_fair_stability.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-repetitions", type=int, default=30)
    parser.add_argument("--seller-fraction", type=float, default=0.70)
    parser.add_argument("--knockoff-repetitions", type=int, default=60)
    parser.add_argument("--stability-inner-repetitions", type=int, default=20)
    parser.add_argument("--shadow-repetitions", type=int, default=8)
    args = parser.parse_args()
    configure()
    started = time.time()

    panel = pd.read_csv(DATA, parse_dates=["month"])
    selection = panel.loc[panel["month"].le("2018-04-01")].reset_index(drop=True)
    train_mask = selection["month"].le("2017-12-01").to_numpy()
    valid_mask = selection["month"].between(
        "2018-01-01", "2018-04-01"
    ).to_numpy()
    y = selection["log_gmv_next_month"].to_numpy(float)
    sellers = selection["seller_id"].to_numpy()
    unique_sellers = np.unique(sellers)

    baseline_prep = RobustPreprocessor(FEATURES).fit(
        selection.loc[train_mask], copula=False
    )
    x_baseline = baseline_prep.transform(selection)
    copula_prep = RobustPreprocessor(FEATURES).fit(selection, copula=True)
    x_copula = copula_prep.transform(selection)

    (
        knock_alpha,
        elastic_alpha,
        elastic_l1,
        stability_alpha,
        xgb_params,
    ) = load_parameters()

    logging.info("Computing full-sample reference sets under the fair protocol")
    references: dict[str, set[int]] = {}
    reference_native_counts: dict[str, int] = {}
    reference_results = {
        "去随机化Knockoff": knockoff_top_set(
            x_copula,
            y,
            knock_alpha,
            SEED + 100,
            args.knockoff_repetitions,
        ),
        "Elastic Net": elastic_top_set(
            x_baseline,
            y,
            elastic_alpha,
            elastic_l1,
        ),
        "稳定性选择": stability_top_set(
            x_baseline,
            y,
            sellers,
            stability_alpha,
            SEED + 300,
            args.stability_inner_repetitions,
        ),
        "影子变量树模型": shadow_top_set(
            x_baseline,
            y,
            SEED + 500,
            args.shadow_repetitions,
        ),
        "XGBoost-SHAP": xgb_shap_top_set(
            x_baseline[train_mask],
            y[train_mask],
            x_baseline[valid_mask],
            xgb_params,
            SEED + 700,
        ),
    }
    reference_rows = []
    for method, (selected, native_count) in reference_results.items():
        references[method] = selected
        reference_native_counts[method] = native_count
        for index in sorted(selected):
            reference_rows.append(
                {
                    "method": method,
                    "feature": FEATURES[index],
                    "label_zh": FEATURE_LABELS_ZH[FEATURES[index]],
                    "matched_budget": BUDGET,
                    "native_count": native_count,
                }
            )
    pd.DataFrame(reference_rows).to_csv(
        RESULTS / "fair_stability_reference_sets.csv", index=False
    )

    records = []
    for outer in range(args.outer_repetitions):
        outer_seed = SEED + outer * 10000
        rng = np.random.default_rng(outer_seed)
        chosen = rng.choice(
            unique_sellers,
            size=max(2, int(len(unique_sellers) * args.seller_fraction)),
            replace=False,
        )
        indices = np.flatnonzero(np.isin(sellers, chosen))
        local_sellers = sellers[indices]
        local_train = train_mask[indices]
        local_valid = valid_mask[indices]

        selections: dict[str, tuple[set[int], int]] = {}
        selections["去随机化Knockoff"] = knockoff_top_set(
            x_copula[indices],
            y[indices],
            knock_alpha,
            outer_seed + 1000,
            args.knockoff_repetitions,
        )
        selections["Elastic Net"] = elastic_top_set(
            x_baseline[indices],
            y[indices],
            elastic_alpha,
            elastic_l1,
        )
        selections["稳定性选择"] = stability_top_set(
            x_baseline[indices],
            y[indices],
            local_sellers,
            stability_alpha,
            outer_seed + 3000,
            args.stability_inner_repetitions,
        )
        selections["影子变量树模型"] = shadow_top_set(
            x_baseline[indices],
            y[indices],
            outer_seed + 5000,
            args.shadow_repetitions,
        )
        selections["XGBoost-SHAP"] = xgb_shap_top_set(
            x_baseline[indices][local_train],
            y[indices][local_train],
            x_baseline[indices][local_valid],
            xgb_params,
            outer_seed + 7000,
        )

        for method in METHODS:
            selected, native_count = selections[method]
            reference = references[method]
            overlap = len(selected & reference)
            records.append(
                {
                    "outer_repetition": outer + 1,
                    "seed": outer_seed,
                    "method": method,
                    "seller_fraction": args.seller_fraction,
                    "selected_sellers": len(chosen),
                    "selected_rows": len(indices),
                    "matched_budget": BUDGET,
                    "native_count": native_count,
                    "overlap": overlap,
                    "jaccard": jaccard(selected, reference),
                    "exact_match": selected == reference,
                    "selected_features": "|".join(
                        FEATURES[index] for index in sorted(selected)
                    ),
                    "selected_labels": "、".join(
                        FEATURE_LABELS_ZH[FEATURES[index]]
                        for index in sorted(selected)
                    ),
                }
            )
        pd.DataFrame(records).to_csv(
            RESULTS / "fair_stability_runs.csv", index=False
        )
        logging.info(
            "Outer repetition %s/%s completed in %.1fs",
            outer + 1,
            args.outer_repetitions,
            time.time() - started,
        )

    runs = pd.DataFrame(records)
    summary = summarize(runs)
    summary.to_csv(RESULTS / "fair_stability_summary.csv", index=False)
    paired = paired_comparisons(runs)
    paired.to_csv(
        RESULTS / "fair_stability_paired_comparison.csv",
        index=False,
    )
    make_figure(summary)

    existing = pd.read_csv(RESULTS / "benchmark_selection_summary.csv")
    old_knockoff = float(
        existing.loc[
            existing["method"].eq("去随机化Knockoff"),
            "resampling_stability",
        ].iloc[0]
    )
    protocol = {
        "seed": SEED,
        "selection_cutoff": "2018-04-01",
        "outer_repetitions": args.outer_repetitions,
        "seller_fraction": args.seller_fraction,
        "paired_seller_subsamples": True,
        "matched_budget": BUDGET,
        "knockoff_repetitions_per_outer_sample": args.knockoff_repetitions,
        "stability_selection_inner_repetitions": (
            args.stability_inner_repetitions
        ),
        "shadow_repetitions_per_outer_sample": args.shadow_repetitions,
        "legacy_knockoff_algorithmic_stability": old_knockoff,
        "legacy_metric_interpretation": (
            "Fixed-data, varying-knockoff ranking stability; retained only as "
            "a separate algorithmic-randomness diagnostic."
        ),
        "elapsed_seconds": time.time() - started,
    }
    (RESULTS / "fair_stability_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Fair stability experiment completed in %.1fs", time.time() - started)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
