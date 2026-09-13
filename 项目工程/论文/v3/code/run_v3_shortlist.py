#!/usr/bin/env python3
"""Build the V3 configurable dimension-selection path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve()
V3_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V1_CODE = V1_DIR / "code"
RESULTS = V3_DIR / "results"
sys.path.insert(0, str(V1_CODE))

from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import RobustPreprocessor, SEED  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of ranked FDR-confirmed dimensions to retain (1-14).",
    )
    parser.add_argument(
        "--contribution-target",
        type=float,
        default=0.90,
        help="Minimum cumulative validation SHAP contribution for default K.",
    )
    parser.add_argument(
        "--rmse-tolerance",
        type=float,
        default=0.005,
        help="Allowed relative validation-RMSE gap from the best K.",
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    q20 = primary.loc[primary["q"].eq(0.20)].copy()
    evidence = q20.copy()
    evidence["strict_ebh_q20"] = evidence["selected_ebh"].astype(bool)
    evidence["stable_90pct"] = evidence["selection_frequency"].ge(0.90)

    panel = pd.read_csv(
        V1_DIR / "data_processed" / "seller_month_panel.csv",
        parse_dates=["month"],
    )
    train = panel.loc[panel["month"].le("2017-12-01")].copy()
    validation = panel.loc[
        panel["month"].between("2018-01-01", "2018-04-01")
    ].copy()
    test = panel.loc[panel["month"].between("2018-05-01", "2018-07-01")].copy()
    preprocessor = RobustPreprocessor(FEATURES).fit(train, copula=False)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    x_test = preprocessor.transform(test)
    target = "log_gmv_next_month"
    y_train = train[target].to_numpy()
    y_validation = validation[target].to_numpy()
    y_test = test[target].to_numpy()
    candidates = []
    for depth, rate, child in [
        (2, 0.04, 10),
        (3, 0.04, 10),
        (3, 0.08, 20),
        (4, 0.04, 20),
    ]:
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
        )
        model.fit(x_train, y_train)
        validation_rmse = mean_squared_error(
            y_validation,
            model.predict(x_validation),
        ) ** 0.5
        candidates.append((validation_rmse, model))
    best_validation_rmse, validation_model = min(
        candidates, key=lambda item: item[0]
    )
    validation_contributions = validation_model.get_booster().predict(
        xgb.DMatrix(x_validation, feature_names=FEATURES),
        pred_contribs=True,
    )[:, :-1]
    validation_shap = pd.DataFrame(
        {
            "feature": FEATURES,
            "validation_mean_abs_shap": np.abs(validation_contributions).mean(
                axis=0
            ),
        }
    )
    evidence = evidence.merge(validation_shap, on="feature", how="left")
    ranked = (
        evidence.loc[evidence["strict_ebh_q20"]]
        .sort_values(
            ["validation_mean_abs_shap", "selection_frequency", "mean_evalue"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    ranked["model_rank"] = np.arange(1, len(ranked) + 1)
    ranked["contribution_share_within_fdr_set"] = (
        ranked["validation_mean_abs_shap"]
        / ranked["validation_mean_abs_shap"].sum()
    )
    ranked["cumulative_contribution"] = ranked[
        "contribution_share_within_fdr_set"
    ].cumsum()

    selected_params = {
        "n_estimators": 550,
        "max_depth": validation_model.max_depth,
        "learning_rate": validation_model.learning_rate,
        "min_child_weight": validation_model.min_child_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": SEED,
    }
    path_rows = []
    predictions = pd.DataFrame(
        {
            "seller_id": test["seller_id"].to_numpy(),
            "month": test["month"].astype(str).to_numpy(),
            "actual_log_gmv": y_test,
        }
    )
    for k in range(1, len(ranked) + 1):
        feature_names = ranked.head(k)["feature"].tolist()
        indices = [FEATURES.index(feature) for feature in feature_names]
        validation_fit = xgb.XGBRegressor(**selected_params).fit(
            x_train[:, indices], y_train
        )
        validation_prediction = validation_fit.predict(x_validation[:, indices])
        validation_rmse = float(
            mean_squared_error(y_validation, validation_prediction) ** 0.5
        )
        final_model = xgb.XGBRegressor(**selected_params).fit(
            np.vstack([x_train, x_validation])[:, indices],
            np.r_[y_train, y_validation],
        )
        test_prediction = final_model.predict(x_test[:, indices])
        test_metrics = metric_row(f"XGBoost-top-{k}", y_test, test_prediction)
        predictions[f"predicted_log_gmv_k{k}"] = test_prediction
        path_rows.append(
            {
                "k": k,
                "validation_rmse_log": validation_rmse,
                **{key: value for key, value in test_metrics.items() if key != "model"},
                "cumulative_contribution": float(
                    ranked.loc[k - 1, "cumulative_contribution"]
                ),
                "features": "|".join(feature_names),
                "labels_zh": "、".join(
                    FEATURE_LABELS_ZH[feature] for feature in feature_names
                ),
            }
        )
    path = pd.DataFrame(path_rows)
    tolerance = args.rmse_tolerance
    contribution_target = args.contribution_target
    if not 0 < contribution_target <= 1:
        raise ValueError("--contribution-target must be in (0, 1]")
    if tolerance < 0:
        raise ValueError("--rmse-tolerance must be non-negative")
    minimum_validation = float(path["validation_rmse_log"].min())
    eligible = path.loc[
        path["cumulative_contribution"].ge(contribution_target)
        & path["validation_rmse_log"].le(
            minimum_validation * (1 + tolerance)
        )
    ]
    if eligible.empty:
        eligible = path.loc[
            path["validation_rmse_log"].le(
                minimum_validation * (1 + tolerance)
            )
        ]
    default_k = int(eligible["k"].min())
    selected_k = default_k if args.k is None else int(args.k)
    if selected_k < 1 or selected_k > len(ranked):
        raise ValueError(f"--k must be between 1 and {len(ranked)}")
    path["recommended_default"] = path["k"].eq(default_k)
    path["selected_by_user"] = path["k"].eq(selected_k)
    path.to_csv(RESULTS / "v3_selection_path.csv", index=False)
    predictions.to_csv(RESULTS / "v3_test_predictions.csv", index=False)
    ranked["recommended_default"] = ranked["model_rank"].le(default_k)
    ranked.to_csv(RESULTS / "v3_ranked_dimensions.csv", index=False)
    evidence = evidence.merge(
        ranked[
            [
                "feature",
                "model_rank",
                "contribution_share_within_fdr_set",
                "cumulative_contribution",
                "recommended_default",
            ]
        ],
        on="feature",
        how="left",
    )
    evidence.to_csv(RESULTS / "v3_all_feature_evidence.csv", index=False)
    shortlist = ranked.head(selected_k).copy()
    shortlist["shortlist_rank"] = shortlist["model_rank"]
    shortlist.to_csv(RESULTS / "v3_shortlist.csv", index=False)

    v1_metrics = pd.read_csv(V1_DIR / "results" / "predictive_model_metrics.csv")
    comparison_k = sorted(set([4, 8, default_k, selected_k, 14]))
    metrics = path.loc[path["k"].isin(comparison_k)].copy()
    metrics["model"] = metrics["k"].map(lambda value: f"XGBoost-K={value}")
    metrics = metrics[
        ["model", "n_test", "rmse_log", "mae_log", "r2_log", "wape_raw"]
    ]
    metrics = pd.concat(
        [
            metrics,
            v1_metrics.loc[
                v1_metrics["model"].isin(["XGBoost", "Naive-current-GMV"])
            ],
        ],
        ignore_index=True,
    )
    metrics.to_csv(RESULTS / "v3_predictive_metrics.csv", index=False)
    default_row = path.loc[path["k"].eq(default_k)].iloc[0]
    full_xgboost_rmse = float(
        v1_metrics.loc[v1_metrics["model"].eq("XGBoost"), "rmse_log"].iloc[0]
    )
    default_rmse = float(default_row["rmse_log"])
    feature_names = shortlist["feature"].tolist()
    metadata = {
        "version": "v3",
        "seed": SEED,
        "selection_parameter": "K",
        "allowed_k": [1, int(len(ranked))],
        "ranking_rule": (
            "Within the q=0.20 e-BH set, rank by validation-set mean absolute "
            "TreeSHAP contribution from a model trained only on the training period"
        ),
        "default_k_rule": (
            "Smallest K with cumulative validation SHAP contribution >= 90% "
            "and validation RMSE within 0.5% of the best K"
        ),
        "performance_tolerance": tolerance,
        "contribution_target": contribution_target,
        "recommended_default_k": default_k,
        "selected_k": selected_k,
        "user_overrode_default_k": args.k is not None,
        "shortlist_count": selected_k,
        "shortlist_features": feature_names,
        "shortlist_labels_zh": [
            FEATURE_LABELS_ZH[feature] for feature in feature_names
        ],
        "formal_fdr_set_count": int(evidence["strict_ebh_q20"].sum()),
        "shortlist_is_operational_not_new_fdr_claim": True,
        "xgboost_parameters": selected_params,
        "default_k_test_rmse": default_rmse,
        "full_xgboost_rmse": full_xgboost_rmse,
        "relative_rmse_change_vs_full_xgboost": (
            default_rmse - full_xgboost_rmse
        )
        / full_xgboost_rmse,
        "best_validation_rmse_across_k": minimum_validation,
        "selection_path_file": "v3_selection_path.csv",
        "panel_sha256": json.loads(
            (V1_DIR / "results" / "data_audit.json").read_text(encoding="utf-8")
        )["panel_sha256"],
    }
    (RESULTS / "v3_shortlist_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
