#!/usr/bin/env python3
"""Run statistical, knockoff, machine-learning, and robustness analyses."""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels.api as sm
import xgboost as xgb
from knockpy.knockoffs import GaussianSampler
from scipy import stats
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from build_panel import FEATURES, FEATURE_LABELS_ZH, V1_DIR


DATA_PATH = V1_DIR / "data_processed" / "seller_month_panel.csv"
RESULTS_DIR = V1_DIR / "results"
FIGURES_DIR = V1_DIR / "figures"
LOG_DIR = V1_DIR / "logs"
KNOCKOFF_RUNS_DIR = RESULTS_DIR / "knockoff_runs"

SEED = 20260823
Q_PRIMARY = 0.20
STATIC_FEATURES = {
    "seller_state_frequency",
    "marketing_origin_frequency",
    "has_marketing_deal",
    "declared_revenue_log",
}
LOG_FEATURES = {
    "avg_price",
    "avg_freight",
    "avg_product_weight_g",
    "avg_product_length_cm",
    "avg_product_height_cm",
    "avg_product_width_cm",
    "avg_product_photos_qty",
    "avg_product_name_length",
    "avg_product_description_length",
    "avg_payment_installments",
    "avg_payment_sequential",
    "avg_delivery_days",
    "avg_approval_days",
    "avg_ship_days",
    "item_count",
    "order_count",
    "category_count",
    "unique_customer_count",
    "buyer_state_diversity",
    "avg_distance_km",
}


def configure() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "analysis.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "Heiti SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_theme(style="whitegrid", font="Arial Unicode MS")
    np.random.seed(SEED)


@dataclass
class RobustPreprocessor:
    features: list[str]
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    median: pd.Series | None = None
    scaler: StandardScaler | None = None
    copula: bool = False

    def _log_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.astype(float).copy()
        for feature in self.features:
            if feature in LOG_FEATURES:
                out[feature] = np.log1p(out[feature].clip(lower=0))
        return out

    def fit(self, frame: pd.DataFrame, copula: bool = False) -> "RobustPreprocessor":
        x = self._log_transform(frame[self.features])
        self.lower = x.quantile(0.01)
        self.upper = x.quantile(0.99)
        x = x.clip(self.lower, self.upper, axis=1)
        self.median = x.median()
        x = x.fillna(self.median)
        self.copula = copula
        x_arr = self._rank_gaussian(x.to_numpy()) if copula else x.to_numpy()
        self.scaler = StandardScaler().fit(x_arr)
        return self

    @staticmethod
    def _rank_gaussian(values: np.ndarray) -> np.ndarray:
        """Randomized distributional transform for ties, with a fixed seed."""
        rng = np.random.default_rng(SEED)
        n, p = values.shape
        transformed = np.empty_like(values, dtype=float)
        for j in range(p):
            order = np.lexsort((rng.random(n), values[:, j]))
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(1, n + 1)
            transformed[:, j] = stats.norm.ppf((ranks - 0.5) / n)
        return transformed

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.lower is None or self.upper is None or self.median is None or self.scaler is None:
            raise RuntimeError("Preprocessor must be fit before transform")
        x = self._log_transform(frame[self.features])
        x = x.clip(self.lower, self.upper, axis=1).fillna(self.median)
        x_arr = self._rank_gaussian(x.to_numpy()) if self.copula else x.to_numpy()
        return self.scaler.transform(x_arr)


def threshold_knockoff_plus(w: np.ndarray, q: float) -> float:
    candidates = np.sort(np.unique(np.abs(w[w != 0])))
    for value in candidates:
        numerator = 1 + np.sum(w <= -value)
        denominator = max(1, int(np.sum(w >= value)))
        if numerator / denominator <= q:
            return float(value)
    return math.inf


def ebh(evalues: np.ndarray, q: float) -> np.ndarray:
    p = len(evalues)
    order = np.argsort(-evalues)
    sorted_values = evalues[order]
    eligible = [k for k in range(1, p + 1) if sorted_values[k - 1] >= p / (q * k)]
    if not eligible:
        return np.array([], dtype=int)
    k = max(eligible)
    cutoff = p / (q * k)
    return np.flatnonzero(evalues >= cutoff)


def make_sampler(x: np.ndarray) -> GaussianSampler:
    sigma = LedoitWolf().fit(x).covariance_
    return GaussianSampler(
        X=x,
        mu=np.mean(x, axis=0),
        Sigma=sigma,
        method="mvr",
    )


def sample_knockoff(sampler: GaussianSampler, seed: int) -> np.ndarray:
    np.random.seed(seed)
    return sampler.sample_knockoffs(check_psd=True)


def choose_lasso_alpha(x: np.ndarray, xk: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([x, xk])
    model = LassoCV(
        alphas=np.logspace(-4, -0.2, 45),
        cv=5,
        max_iter=15000,
        n_jobs=-1,
        random_state=SEED,
    ).fit(design, y)
    return float(model.alpha_)


def lasso_w(x: np.ndarray, xk: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    model = Lasso(alpha=alpha, max_iter=15000, selection="cyclic").fit(np.column_stack([x, xk]), y)
    p = x.shape[1]
    return np.abs(model.coef_[:p]) - np.abs(model.coef_[p:])


def xgb_w(x: np.ndarray, xk: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = x.shape[1]
    pair_swap = rng.integers(0, 2, p).astype(bool)
    left, right = x.copy(), xk.copy()
    left[:, pair_swap], right[:, pair_swap] = xk[:, pair_swap], x[:, pair_swap]
    design = np.column_stack([left, right])
    model = xgb.XGBRegressor(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=seed,
    ).fit(design, y)
    imp_left = model.feature_importances_[:p]
    imp_right = model.feature_importances_[p:]
    real = np.where(pair_swap, imp_right, imp_left)
    knock = np.where(pair_swap, imp_left, imp_right)
    return real - knock


def repeated_knockoffs(
    frame: pd.DataFrame,
    target: str,
    generator: str,
    repetitions: int,
    q_values: tuple[float, ...] = (0.10, 0.20, 0.30),
    stat_model: str = "lasso",
    seed_offset: int = 0,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    logging.info(
        "Knockoff scenario target=%s generator=%s stat=%s n=%s M=%s",
        target,
        generator,
        stat_model,
        len(frame),
        repetitions,
    )
    prep = RobustPreprocessor(FEATURES).fit(frame, copula=generator == "copula")
    x = prep.transform(frame)
    y = frame[target].to_numpy(dtype=float)
    y = (y - y.mean()) / y.std(ddof=0)
    sampler = make_sampler(x)
    first_xk = sample_knockoff(sampler, SEED + seed_offset)
    alpha = choose_lasso_alpha(x, first_xk, y) if stat_model == "lasso" else np.nan
    w_runs = []
    for rep in range(repetitions):
        seed = SEED + seed_offset + rep
        xk = first_xk if rep == 0 else sample_knockoff(sampler, seed)
        w = lasso_w(x, xk, y, alpha) if stat_model == "lasso" else xgb_w(x, xk, y, seed)
        w_runs.append(w)
    w_matrix = np.vstack(w_runs)
    rows = []
    selected_by_q = {}
    alpha_kn_by_q = {}
    for q in q_values:
        # Ren and Barber (2024) recommend alpha_kn = alpha_eBH / 2 for
        # repeated knockoffs. The final FDR level is determined by e-BH.
        alpha_kn = q / 2
        alpha_kn_by_q[str(q)] = alpha_kn
        e_runs = np.zeros_like(w_matrix)
        selected_runs = np.zeros_like(w_matrix, dtype=bool)
        thresholds = []
        for rep, w in enumerate(w_matrix):
            threshold = threshold_knockoff_plus(w, alpha_kn)
            thresholds.append(threshold)
            if np.isfinite(threshold):
                selected = w >= threshold
                denominator = 1 + np.sum(w <= -threshold)
                e_runs[rep, selected] = len(FEATURES) / denominator
                selected_runs[rep] = selected
        avg_e = e_runs.mean(axis=0)
        final_indices = ebh(avg_e, q)
        selected_by_q[str(q)] = [FEATURES[i] for i in final_indices]
        for j, feature in enumerate(FEATURES):
            rows.append({
                "target": target,
                "generator": generator,
                "stat_model": stat_model,
                "q": q,
                "alpha_kn": alpha_kn,
                "feature": feature,
                "label_zh": FEATURE_LABELS_ZH[feature],
                "mean_w": float(w_matrix[:, j].mean()),
                "median_w": float(np.median(w_matrix[:, j])),
                "positive_w_rate": float((w_matrix[:, j] > 0).mean()),
                "selection_frequency": float(selected_runs[:, j].mean()),
                "mean_evalue": float(avg_e[j]),
                "selected_ebh": bool(j in final_indices),
            })
    meta = {
        "target": target,
        "generator": generator,
        "stat_model": stat_model,
        "n": len(frame),
        "p": len(FEATURES),
        "repetitions": repetitions,
        "lasso_alpha": None if np.isnan(alpha) else alpha,
        "alpha_kn_by_q": alpha_kn_by_q,
        "selected_by_q": selected_by_q,
    }
    return pd.DataFrame(rows), meta, w_matrix


def save_w_runs(name: str, w_matrix: np.ndarray) -> None:
    KNOCKOFF_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(w_matrix, columns=FEATURES)
    frame.insert(0, "repetition", np.arange(1, len(frame) + 1))
    frame.to_csv(KNOCKOFF_RUNS_DIR / f"{name}_w.csv", index=False)


def knockoff_diagnostics(frame: pd.DataFrame, generator: str) -> dict:
    prep = RobustPreprocessor(FEATURES).fit(frame, copula=generator == "copula")
    x = prep.transform(frame)
    xk = sample_knockoff(make_sampler(x), SEED + 9000)
    sigma_x = np.cov(x, rowvar=False)
    sigma_k = np.cov(xk, rowvar=False)
    cross = np.cov(x.T, xk.T)[: len(FEATURES), len(FEATURES) :]
    ks = [stats.ks_2samp(x[:, j], xk[:, j]).statistic for j in range(x.shape[1])]
    pair_corr = [
        np.corrcoef(x[:, j], xk[:, j])[0, 1]
        for j in range(x.shape[1])
        if np.std(x[:, j]) > 0 and np.std(xk[:, j]) > 0
    ]
    return {
        "generator": generator,
        "mean_marginal_ks": float(np.mean(ks)),
        "max_marginal_ks": float(np.max(ks)),
        "covariance_relative_error": float(np.linalg.norm(sigma_x - sigma_k) / np.linalg.norm(sigma_x)),
        "cross_covariance_asymmetry": float(np.linalg.norm(cross - cross.T) / np.linalg.norm(sigma_x)),
        "mean_original_knockoff_correlation": float(np.mean(pair_corr)),
        "min_original_knockoff_correlation": float(np.min(pair_corr)),
        "max_original_knockoff_correlation": float(np.max(pair_corr)),
    }


def descriptive_outputs(panel: pd.DataFrame) -> None:
    rows = []
    for feature in FEATURES:
        s = panel[feature]
        rows.append({
            "feature": feature,
            "label_zh": FEATURE_LABELS_ZH[feature],
            "n": int(s.notna().sum()),
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
            "spearman_next_gmv": float(s.corr(panel["log_gmv_next_month"], method="spearman")),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "descriptive_statistics.csv", index=False)
    monthly = panel.groupby("year_month", as_index=False).agg(
        gmv=("gmv", "sum"),
        seller_months=("seller_id", "size"),
        active_sellers=("seller_id", "nunique"),
        orders=("order_count", "sum"),
        next_month_gmv=("gmv_next_month", "sum"),
    )
    monthly.to_csv(RESULTS_DIR / "monthly_summary.csv", index=False)
    target_summary = panel[["gmv", "gmv_next_month", "log_gmv", "log_gmv_next_month", "aov_next_month"]].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).T
    target_summary.to_csv(RESULTS_DIR / "target_summary.csv")


def metric_row(name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    raw_y = np.expm1(y)
    raw_pred = np.maximum(0, np.expm1(pred))
    return {
        "model": name,
        "n_test": len(y),
        "rmse_log": float(mean_squared_error(y, pred) ** 0.5),
        "mae_log": float(mean_absolute_error(y, pred)),
        "r2_log": float(r2_score(y, pred)),
        "wape_raw": float(np.abs(raw_y - raw_pred).sum() / max(raw_y.sum(), 1e-12)),
    }


def predictive_models(panel: pd.DataFrame, selected_features: list[str]) -> dict:
    train = panel.loc[panel["month"] <= "2017-12-01"].copy()
    valid = panel.loc[panel["month"].between("2018-01-01", "2018-04-01")].copy()
    test = panel.loc[panel["month"].between("2018-05-01", "2018-07-01")].copy()
    target = "log_gmv_next_month"
    prep = RobustPreprocessor(FEATURES).fit(train, copula=False)
    x_train, x_valid, x_test = prep.transform(train), prep.transform(valid), prep.transform(test)
    y_train, y_valid, y_test = (x[target].to_numpy() for x in (train, valid, test))
    rows = [metric_row("Naive-current-GMV", y_test, test["log_gmv"].to_numpy())]

    ridge_candidates = [0.1, 1.0, 10.0, 100.0]
    ridge = min(
        (Ridge(alpha=a).fit(x_train, y_train) for a in ridge_candidates),
        key=lambda model: mean_squared_error(y_valid, model.predict(x_valid)),
    )
    ridge_best = ridge.alpha
    ridge.fit(np.vstack([x_train, x_valid]), np.r_[y_train, y_valid])
    rows.append(metric_row("Ridge", y_test, ridge.predict(x_test)))

    lasso = LassoCV(alphas=np.logspace(-4, 0, 60), cv=5, max_iter=15000, n_jobs=-1).fit(x_train, y_train)
    lasso_alpha = float(lasso.alpha_)
    lasso = Lasso(alpha=lasso_alpha, max_iter=15000).fit(np.vstack([x_train, x_valid]), np.r_[y_train, y_valid])
    rows.append(metric_row("Lasso", y_test, lasso.predict(x_test)))

    extra_candidates = [
        ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=SEED),
        ExtraTreesRegressor(n_estimators=500, min_samples_leaf=8, max_features=1.0, n_jobs=-1, random_state=SEED),
    ]
    extra = min(
        (model.fit(x_train, y_train) for model in extra_candidates),
        key=lambda model: mean_squared_error(y_valid, model.predict(x_valid)),
    )
    extra_params = {"min_samples_leaf": extra.min_samples_leaf, "max_features": extra.max_features}
    extra.fit(np.vstack([x_train, x_valid]), np.r_[y_train, y_valid])
    rows.append(metric_row("ExtraTrees", y_test, extra.predict(x_test)))

    xgb_candidates = []
    for depth, rate, child in [(2, 0.04, 10), (3, 0.04, 10), (3, 0.08, 20), (4, 0.04, 20)]:
        xgb_candidates.append(xgb.XGBRegressor(
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
        ))
    xgb_model = min(
        (model.fit(x_train, y_train) for model in xgb_candidates),
        key=lambda model: mean_squared_error(y_valid, model.predict(x_valid)),
    )
    xgb_params = {
        "max_depth": xgb_model.max_depth,
        "learning_rate": xgb_model.learning_rate,
        "min_child_weight": xgb_model.min_child_weight,
    }
    xgb_model.fit(np.vstack([x_train, x_valid]), np.r_[y_train, y_valid])
    pred_xgb = xgb_model.predict(x_test)
    rows.append(metric_row("XGBoost", y_test, pred_xgb))

    mlp_candidates = [
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001, early_stopping=True, max_iter=500, random_state=SEED),
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, early_stopping=True, max_iter=500, random_state=SEED),
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.01, early_stopping=True, max_iter=500, random_state=SEED),
    ]
    mlp = min(
        (model.fit(x_train, y_train) for model in mlp_candidates),
        key=lambda model: mean_squared_error(y_valid, model.predict(x_valid)),
    )
    mlp_params = {"hidden_layer_sizes": mlp.hidden_layer_sizes, "alpha": mlp.alpha}
    mlp.fit(np.vstack([x_train, x_valid]), np.r_[y_train, y_valid])
    rows.append(metric_row("MLP", y_test, mlp.predict(x_test)))

    selected = [f for f in selected_features if f in FEATURES]
    if selected:
        indices = [FEATURES.index(f) for f in selected]
        selected_model = clone(xgb_model).fit(
            np.vstack([x_train, x_valid])[:, indices], np.r_[y_train, y_valid]
        )
        pred_selected = selected_model.predict(x_test[:, indices])
        rows.append(metric_row("XGBoost-Knockoff-selected", y_test, pred_selected))

    metrics = pd.DataFrame(rows).sort_values("rmse_log")
    metrics.to_csv(RESULTS_DIR / "predictive_model_metrics.csv", index=False)
    predictions = test[["seller_id", "year_month", "gmv_next_month", target]].copy()
    predictions["pred_log_xgboost"] = pred_xgb
    predictions["pred_gmv_xgboost"] = np.maximum(0, np.expm1(pred_xgb))
    predictions.to_csv(RESULTS_DIR / "xgboost_test_predictions.csv", index=False)

    dmatrix = xgb.DMatrix(x_test, feature_names=FEATURES)
    contributions = xgb_model.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]
    shap_table = pd.DataFrame({
        "feature": FEATURES,
        "label_zh": [FEATURE_LABELS_ZH[f] for f in FEATURES],
        "mean_abs_shap": np.abs(contributions).mean(axis=0),
        "mean_shap": contributions.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_table.to_csv(RESULTS_DIR / "xgboost_shap_importance.csv", index=False)

    perm = permutation_importance(
        xgb_model,
        x_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    pd.DataFrame({
        "feature": FEATURES,
        "label_zh": [FEATURE_LABELS_ZH[f] for f in FEATURES],
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_sd": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False).to_csv(
        RESULTS_DIR / "xgboost_permutation_importance.csv", index=False
    )

    metadata = {
        "train_n": len(train),
        "validation_n": len(valid),
        "test_n": len(test),
        "train_period": [train["year_month"].min(), train["year_month"].max()],
        "validation_period": [valid["year_month"].min(), valid["year_month"].max()],
        "test_period": [test["year_month"].min(), test["year_month"].max()],
        "ridge_alpha": ridge_best,
        "lasso_alpha": lasso_alpha,
        "extra_trees": extra_params,
        "xgboost": xgb_params,
        "mlp": mlp_params,
        "pretest_selected_features": selected,
    }
    return metadata


def two_way_demean(values: pd.DataFrame, seller: pd.Series, month: pd.Series, iterations: int = 30) -> pd.DataFrame:
    out = values.astype(float).copy()
    for _ in range(iterations):
        previous = out.to_numpy().copy()
        out -= out.groupby(seller).transform("mean")
        out -= out.groupby(month).transform("mean")
        if np.max(np.abs(out.to_numpy() - previous)) < 1e-9:
            break
    return out


def fixed_effects(panel: pd.DataFrame) -> None:
    prep = RobustPreprocessor(FEATURES).fit(panel, copula=False)
    x = pd.DataFrame(prep.transform(panel), columns=FEATURES, index=panel.index)
    model_frame = pd.concat([panel[["seller_id", "year_month", "log_gmv_next_month"]], x], axis=1)
    dynamic_features = [feature for feature in FEATURES if feature not in STATIC_FEATURES]
    dm = two_way_demean(
        model_frame[["log_gmv_next_month"] + dynamic_features],
        model_frame["seller_id"],
        model_frame["year_month"],
    )
    fit = sm.OLS(dm["log_gmv_next_month"], dm[dynamic_features]).fit(
        cov_type="cluster", cov_kwds={"groups": model_frame["seller_id"], "use_correction": True}
    )
    ci = fit.conf_int()
    table = pd.DataFrame({
        "feature": dynamic_features,
        "label_zh": [FEATURE_LABELS_ZH[f] for f in dynamic_features],
        "coefficient": fit.params,
        "cluster_se": fit.bse,
        "t_value": fit.tvalues,
        "p_value": fit.pvalues,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
    }).reset_index(drop=True)
    table.to_csv(RESULTS_DIR / "two_way_fixed_effects.csv", index=False)
    with (RESULTS_DIR / "fixed_effects_model.txt").open("w", encoding="utf-8") as handle:
        handle.write(fit.summary().as_text())


def calibration_simulation(
    panel: pd.DataFrame,
    repetitions: int = 30,
    knockoff_repetitions: int = 20,
) -> dict:
    prep = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    x = prep.transform(panel)
    true_idx = np.array([FEATURES.index(f) for f in [
        "avg_price", "avg_review_score", "item_count", "category_count", "avg_delivery_days"
    ]])
    beta = np.zeros(len(FEATURES))
    beta[true_idx] = np.array([0.8, 0.55, 0.9, 0.65, -0.55])
    signal = x @ beta
    noise_sd = signal.std() / 2.0
    fdp, power, discoveries = [], [], []
    base_fdp, base_power = [], []
    sampler = make_sampler(x)
    for rep in range(repetitions):
        rng = np.random.default_rng(SEED + 30000 + rep)
        y = signal + rng.normal(0, noise_sd, len(x))
        e_runs = np.zeros((knockoff_repetitions, len(FEATURES)))
        first_w = None
        for knockoff_rep in range(knockoff_repetitions):
            seed = SEED + 40000 + rep * knockoff_repetitions + knockoff_rep
            xk = sample_knockoff(sampler, seed)
            alpha = choose_lasso_alpha(x, xk, y) if rep == 0 and knockoff_rep == 0 else fixed_alpha
            if rep == 0 and knockoff_rep == 0:
                fixed_alpha = alpha
            w = lasso_w(x, xk, y, fixed_alpha)
            if knockoff_rep == 0:
                first_w = w
            threshold = threshold_knockoff_plus(w, Q_PRIMARY / 2)
            if np.isfinite(threshold):
                selected_run = w >= threshold
                denominator = 1 + np.sum(w <= -threshold)
                e_runs[knockoff_rep, selected_run] = len(FEATURES) / denominator

        selected = ebh(e_runs.mean(axis=0), Q_PRIMARY)
        false = len(set(selected) - set(true_idx))
        true = len(set(selected) & set(true_idx))
        fdp.append(false / max(len(selected), 1))
        power.append(true / len(true_idx))
        discoveries.append(len(selected))

        base_threshold = threshold_knockoff_plus(first_w, Q_PRIMARY)
        base_selected = (
            np.flatnonzero(first_w >= base_threshold)
            if np.isfinite(base_threshold)
            else np.array([], dtype=int)
        )
        base_false = len(set(base_selected) - set(true_idx))
        base_true = len(set(base_selected) & set(true_idx))
        base_fdp.append(base_false / max(len(base_selected), 1))
        base_power.append(base_true / len(true_idx))
    result = {
        "method": "derandomized_knockoffs_eBH",
        "repetitions": repetitions,
        "knockoff_repetitions_per_dataset": knockoff_repetitions,
        "alpha_ebh": Q_PRIMARY,
        "alpha_kn": Q_PRIMARY / 2,
        "true_features": [FEATURES[i] for i in true_idx],
        "mean_fdp": float(np.mean(fdp)),
        "probability_fdp_above_q": float(np.mean(np.array(fdp) > Q_PRIMARY)),
        "mean_power": float(np.mean(power)),
        "mean_discoveries": float(np.mean(discoveries)),
        "single_knockoff_mean_fdp": float(np.mean(base_fdp)),
        "single_knockoff_mean_power": float(np.mean(base_power)),
    }
    with (RESULTS_DIR / "simulation_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def make_figures(panel: pd.DataFrame, primary: pd.DataFrame) -> None:
    monthly = pd.read_csv(RESULTS_DIR / "monthly_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(monthly["year_month"], monthly["gmv"] / 1e6, marker="o", color="#176B87", linewidth=2)
    ax.set_ylabel("GMV（百万雷亚尔）")
    ax.set_xlabel("月份")
    ax.set_title("Olist 月度成交额（主分析期）")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_monthly_gmv.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(panel["log_gmv_next_month"], bins=40, color="#176B87", alpha=0.85, edgecolor="white")
    ax.set_xlabel("log(1 + 次月GMV)")
    ax.set_ylabel("卖家-月观测数")
    ax.set_title("次月GMV目标变量分布")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_target_distribution.png")
    plt.close(fig)

    view = primary.loc[primary["q"].eq(Q_PRIMARY)].sort_values("mean_evalue", ascending=True)
    colors = np.where(view["selected_ebh"], "#D1495B", "#5B7083")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(view["label_zh"], view["mean_evalue"], color=colors)
    ax.set_xlabel("平均 e-value")
    ax.set_title("去随机化 Knockoff 变量选择（q=0.20）")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_knockoff_evalues.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(view["label_zh"], view["selection_frequency"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("单次 Knockoff 入选频率")
    ax.set_title("变量入选稳定性（αeBH=0.20，αkn=0.10）")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_selection_frequency.png")
    plt.close(fig)

    metrics = pd.read_csv(RESULTS_DIR / "predictive_model_metrics.csv").sort_values("rmse_log", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(metrics["model"], metrics["rmse_log"], color="#2A9D8F")
    ax.set_xlabel("测试集 RMSE（对数尺度，越低越好）")
    ax.set_title("时间外推预测性能")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_model_performance.png")
    plt.close(fig)

    shap_table = pd.read_csv(RESULTS_DIR / "xgboost_shap_importance.csv").head(15).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shap_table["label_zh"], shap_table["mean_abs_shap"], color="#E9C46A")
    ax.set_xlabel("平均 |SHAP|（对数GMV）")
    ax.set_title("XGBoost 前15项特征重要性")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_xgboost_shap.png")
    plt.close(fig)

    preds = pd.read_csv(RESULTS_DIR / "xgboost_test_predictions.csv")
    cap = np.quantile(preds["gmv_next_month"], 0.99)
    shown = preds.loc[preds["gmv_next_month"] <= cap]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(shown["gmv_next_month"], shown["pred_gmv_xgboost"], s=10, alpha=0.25, color="#176B87")
    limit = max(shown["gmv_next_month"].max(), shown["pred_gmv_xgboost"].max())
    ax.plot([0, limit], [0, limit], linestyle="--", color="#D1495B")
    ax.set_xlabel("实际次月GMV（截尾至P99）")
    ax.set_ylabel("预测次月GMV")
    ax.set_title("XGBoost 时间外测试：实际值与预测值")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_actual_vs_predicted.png")
    plt.close(fig)


def main() -> None:
    configure()
    started = time.time()
    panel = pd.read_csv(DATA_PATH, parse_dates=["month"])
    logging.info("Loaded panel n=%s p=%s", len(panel), len(FEATURES))
    descriptive_outputs(panel)

    all_results = []
    metadata = []
    primary, primary_meta, primary_w = repeated_knockoffs(
        panel, "log_gmv_next_month", "copula", repetitions=60, seed_offset=1000
    )
    save_w_runs("primary_copula_lasso", primary_w)
    all_results.append(primary)
    metadata.append(primary_meta)
    gaussian, gaussian_meta, gaussian_w = repeated_knockoffs(
        panel, "log_gmv_next_month", "gaussian", repetitions=40, seed_offset=2000
    )
    save_w_runs("gaussian_lasso", gaussian_w)
    all_results.append(gaussian)
    metadata.append(gaussian_meta)
    same_month, same_meta, same_month_w = repeated_knockoffs(
        panel, "log_gmv", "copula", repetitions=40, q_values=(0.20,), seed_offset=3000
    )
    save_w_runs("same_month_gmv", same_month_w)
    all_results.append(same_month)
    metadata.append(same_meta)
    trimmed_frame = panel.loc[panel["month"].between("2017-03-01", "2018-05-01")]
    trimmed, trimmed_meta, trimmed_w = repeated_knockoffs(
        trimmed_frame, "log_gmv_next_month", "copula", repetitions=40, q_values=(0.20,), seed_offset=4000
    )
    save_w_runs("trimmed_period", trimmed_w)
    all_results.append(trimmed.assign(generator="copula_trimmed_period"))
    trimmed_meta["generator"] = "copula_trimmed_period"
    metadata.append(trimmed_meta)
    aov, aov_meta, aov_w = repeated_knockoffs(
        panel, "log_aov_next_month", "copula", repetitions=40, q_values=(0.20,), seed_offset=5000
    )
    save_w_runs("next_month_aov", aov_w)
    all_results.append(aov)
    metadata.append(aov_meta)
    ai, ai_meta, ai_w = repeated_knockoffs(
        panel,
        "log_gmv_next_month",
        "copula",
        repetitions=25,
        q_values=(0.20,),
        stat_model="xgboost",
        seed_offset=6000,
    )
    save_w_runs("copula_xgboost", ai_w)
    all_results.append(ai.assign(generator="copula_ai_xgboost"))
    ai_meta["generator"] = "copula_ai_xgboost"
    metadata.append(ai_meta)

    pretest = panel.loc[panel["month"] <= "2018-04-01"]
    pretest_result, pretest_meta, pretest_w = repeated_knockoffs(
        pretest, "log_gmv_next_month", "copula", repetitions=40, q_values=(0.20,), seed_offset=7000
    )
    save_w_runs("pretest_copula_lasso", pretest_w)
    all_results.append(pretest_result.assign(generator="copula_pretest"))
    pretest_meta["generator"] = "copula_pretest"
    metadata.append(pretest_meta)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "knockoff_all_scenarios.csv", index=False)
    primary.to_csv(RESULTS_DIR / "knockoff_primary.csv", index=False)
    with (RESULTS_DIR / "knockoff_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    primary_selected = primary.loc[
        primary["q"].eq(Q_PRIMARY) & primary["selected_ebh"], "feature"
    ].tolist()
    primary_stable = primary.loc[
        primary["q"].eq(Q_PRIMARY) & primary["selection_frequency"].ge(0.90), "feature"
    ].tolist()
    pretest_selected = pretest_result.loc[
        pretest_result["q"].eq(Q_PRIMARY) & pretest_result["selected_ebh"], "feature"
    ].tolist()
    model_meta = predictive_models(panel, pretest_selected)
    fixed_effects(panel)
    diagnostics = [knockoff_diagnostics(panel, name) for name in ("gaussian", "copula")]
    with (RESULTS_DIR / "knockoff_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
    calibration = calibration_simulation(panel)
    make_figures(panel, primary)

    reproducibility = {
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": sm.__version__,
        "xgboost": xgb.__version__,
        "primary_selected_features": primary_selected,
        "primary_stable_features_at_90pct": primary_stable,
        "pretest_selected_features": pretest_selected,
        "model_metadata": model_meta,
        "simulation_calibration": calibration,
        "elapsed_seconds": time.time() - started,
    }
    with (RESULTS_DIR / "reproducibility.json").open("w", encoding="utf-8") as handle:
        json.dump(reproducibility, handle, ensure_ascii=False, indent=2, default=str)
    logging.info("Primary selected: %s", primary_selected)
    logging.info("Completed in %.1f seconds", time.time() - started)


if __name__ == "__main__":
    main()
