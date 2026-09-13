#!/usr/bin/env python3
"""Run the V4 clustered, group, adversarial, and deep-importance analyses."""

from __future__ import annotations

import json
import logging
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
V4_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
V2_DIR = PROJECT_DIR / "论文" / "v2"
sys.path.insert(0, str(V1_DIR / "code"))
sys.path.insert(0, str(V2_DIR / "code"))

from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import (  # noqa: E402
    RobustPreprocessor,
    ebh,
    sample_knockoff,
    threshold_knockoff_plus,
)
from run_deep_analysis import (  # noqa: E402
    DeepKnockoffGenerator,
    PairwiseCompetitiveMLP,
    covariance,
    generate_torch,
    get_device,
    mixed_rbf_mmd,
    quick_generator_score,
    rbf_mmd_permutation,
    set_seed,
    swap_classifier_auc,
)
from knockpy.knockoffs import GaussianSampler  # noqa: E402


RESULTS = V4_DIR / "results"
FIGURES = V4_DIR / "figures"
MODELS = V4_DIR / "models"
LOGS = V4_DIR / "logs"
SEED = 20260823
Q_VALUES = (0.10, 0.20, 0.30)

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "STHeiti",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


GROUP_DEFINITIONS = {
    "price_cost": ["avg_price", "avg_freight"],
    "physical_size": [
        "avg_product_weight_g",
        "avg_product_length_cm",
        "avg_product_height_cm",
        "avg_product_width_cm",
    ],
    "content": [
        "avg_product_photos_qty",
        "avg_product_name_length",
        "avg_product_description_length",
    ],
    "payment_process": [
        "avg_payment_installments",
        "avg_payment_sequential",
    ],
    "payment_method": [
        "credit_card_ratio",
        "boleto_ratio",
        "voucher_ratio",
        "debit_card_ratio",
    ],
    "reputation": ["avg_review_score"],
    "fulfillment": [
        "avg_delivery_days",
        "avg_approval_days",
        "avg_ship_days",
        "avg_estimate_gap_days",
    ],
    "transaction_scale": [
        "item_count",
        "order_count",
        "category_count",
        "unique_customer_count",
    ],
    "market_concentration": ["dominant_customer_state_share"],
    "market_reach": [
        "buyer_state_diversity",
        "interstate_ratio",
        "avg_distance_km",
    ],
    "seller_geography": ["seller_state_frequency"],
    "marketing": ["marketing_origin_frequency", "has_marketing_deal"],
}

GROUP_LABELS_ZH = {
    "price_cost": "价格成本",
    "physical_size": "商品物理属性",
    "content": "商品内容",
    "payment_process": "支付过程",
    "payment_method": "支付方式",
    "reputation": "口碑",
    "fulfillment": "履约",
    "transaction_scale": "交易规模",
    "market_concentration": "市场集中度",
    "market_reach": "市场覆盖",
    "seller_geography": "卖家地域",
    "marketing": "营销",
}


@dataclass
class V4Config:
    generator_epochs: int = 100
    generator_patience: int = 14
    clustered_repetitions: int = 40
    group_repetitions: int = 60
    generator_repetitions: int = 30
    grip_repetitions: int = 20
    grip_epochs: int = 90
    deep_kernel_epochs: int = 45
    diagnostic_permutations: int = 199


def configure_logging() -> None:
    for directory in (RESULTS, FIGURES, MODELS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "v4_analysis.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def make_sampler(x: np.ndarray, groups: np.ndarray | None = None) -> GaussianSampler:
    sigma = LedoitWolf().fit(x).covariance_
    return GaussianSampler(
        X=x,
        mu=np.mean(x, axis=0),
        Sigma=sigma,
        groups=groups,
        method="mvr",
    )


def randomized_rank_gaussian(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, p = values.shape
    transformed = np.empty((n, p), dtype=float)
    for column in range(p):
        source = values[:, column]
        finite = np.isfinite(source)
        fill = np.nanmedian(source[finite]) if finite.any() else 0.0
        source = np.where(finite, source, fill)
        order = np.lexsort((rng.random(n), source))
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        transformed[:, column] = stats.norm.ppf((ranks - 0.5) / n)
    return StandardScaler().fit_transform(transformed)


def choose_alpha(x: np.ndarray, xk: np.ndarray, y: np.ndarray) -> float:
    model = LassoCV(
        alphas=np.logspace(-4, -0.2, 38),
        cv=5,
        max_iter=15000,
        n_jobs=-1,
        random_state=SEED,
    ).fit(np.hstack([x, xk]), y)
    return float(model.alpha_)


def lasso_w(x: np.ndarray, xk: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    model = Lasso(alpha=alpha, max_iter=15000, tol=1e-5).fit(
        np.hstack([x, xk]), y
    )
    p = x.shape[1]
    return np.abs(model.coef_[:p]) - np.abs(model.coef_[p:])


def summarize_w(
    w_matrix: np.ndarray,
    feature_names: list[str],
    labels: dict[str, str],
    method: str,
    q_values: tuple[float, ...] = Q_VALUES,
) -> pd.DataFrame:
    p = len(feature_names)
    rows = []
    for q in q_values:
        alpha_kn = q / 2
        e_runs = np.zeros_like(w_matrix)
        selected_runs = np.zeros_like(w_matrix, dtype=bool)
        for repetition, w in enumerate(w_matrix):
            threshold = threshold_knockoff_plus(w, alpha_kn)
            if np.isfinite(threshold):
                selected = w >= threshold
                denominator = 1 + np.sum(w <= -threshold)
                e_runs[repetition, selected] = p / denominator
                selected_runs[repetition] = selected
        mean_e = e_runs.mean(axis=0)
        selected_final = set(ebh(mean_e, q).tolist())
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "method": method,
                    "q": q,
                    "alpha_kn": alpha_kn,
                    "feature": feature,
                    "label_zh": labels[feature],
                    "mean_w": float(w_matrix[:, index].mean()),
                    "positive_w_rate": float((w_matrix[:, index] > 0).mean()),
                    "selection_frequency": float(
                        selected_runs[:, index].mean()
                    ),
                    "mean_evalue": float(mean_e[index]),
                    "selected_ebh": index in selected_final,
                }
            )
    return pd.DataFrame(rows)


class EmpiricalMarginalCalibrator:
    """Fit raw-generator to target quantile maps on training X only."""

    def __init__(self) -> None:
        self.raw_sorted: list[np.ndarray] = []
        self.target_sorted: list[np.ndarray] = []

    def fit(
        self, raw_training: np.ndarray, target_training: np.ndarray
    ) -> "EmpiricalMarginalCalibrator":
        self.raw_sorted = [
            np.sort(raw_training[:, column])
            for column in range(raw_training.shape[1])
        ]
        self.target_sorted = [
            np.sort(target_training[:, column])
            for column in range(target_training.shape[1])
        ]
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        calibrated = np.empty_like(raw)
        for column, (raw_reference, target_reference) in enumerate(
            zip(self.raw_sorted, self.target_sorted)
        ):
            calibrated[:, column] = np.interp(
                raw[:, column],
                raw_reference,
                target_reference,
                left=target_reference[0],
                right=target_reference[-1],
            )
        return calibrated


def run_clustered_knockoffs(
    panel: pd.DataFrame, config: V4Config
) -> tuple[pd.DataFrame, dict]:
    seller_means = panel.groupby("seller_id", as_index=False).agg(
        {**{feature: "mean" for feature in FEATURES}, "log_gmv_next_month": "mean"}
    )
    between_preprocessor = RobustPreprocessor(FEATURES).fit(
        seller_means, copula=True
    )
    x_between = between_preprocessor.transform(seller_means)
    y_between = seller_means["log_gmv_next_month"].to_numpy(float)
    y_between = (y_between - y_between.mean()) / y_between.std()

    seller_feature_means = panel.groupby("seller_id")[FEATURES].transform("mean")
    deviations = panel[FEATURES] - seller_feature_means
    within_features = [
        feature
        for feature in FEATURES
        if deviations[feature].std(skipna=True) > 1e-8
    ]
    x_within = randomized_rank_gaussian(
        deviations[within_features].to_numpy(float), SEED + 21000
    )
    target = panel["log_gmv_next_month"]
    y_within = (
        target - target.groupby(panel["seller_id"]).transform("mean")
    ).to_numpy(float)
    y_within = (y_within - np.nanmean(y_within)) / np.nanstd(y_within)

    outputs = []
    metadata = {}
    for level, x, y, names, seed_offset in [
        ("between_seller", x_between, y_between, FEATURES, 22000),
        ("within_seller", x_within, y_within, within_features, 23000),
    ]:
        sampler = make_sampler(x)
        first = sample_knockoff(sampler, SEED + seed_offset)
        alpha = choose_alpha(x, first, y)
        w_runs = []
        for repetition in range(config.clustered_repetitions):
            xk = (
                first
                if repetition == 0
                else sample_knockoff(
                    sampler, SEED + seed_offset + repetition
                )
            )
            w_runs.append(lasso_w(x, xk, y, alpha))
        w_matrix = np.asarray(w_runs)
        pd.DataFrame(w_matrix, columns=names).to_csv(
            RESULTS / f"clustered_{level}_w.csv",
            index_label="repetition",
        )
        outputs.append(
            summarize_w(
                w_matrix,
                names,
                FEATURE_LABELS_ZH,
                f"clustered_{level}",
            )
        )
        metadata[level] = {
            "n": len(x),
            "p": len(names),
            "repetitions": config.clustered_repetitions,
            "lasso_alpha": alpha,
        }
        logging.info("Clustered level %s completed", level)
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(RESULTS / "clustered_knockoff_selection.csv", index=False)
    return result, metadata


def group_w(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Lasso(alpha=alpha, max_iter=15000, tol=1e-5).fit(
        np.hstack([x, xk]), y
    )
    p = x.shape[1]
    output = []
    for group_id in np.unique(groups):
        indices = np.flatnonzero(groups == group_id)
        real = np.linalg.norm(model.coef_[indices])
        knockoff = np.linalg.norm(model.coef_[p + indices])
        output.append(real - knockoff)
    return np.asarray(output)


def run_group_knockoff(
    panel: pd.DataFrame, config: V4Config
) -> tuple[pd.DataFrame, dict]:
    group_names = list(GROUP_DEFINITIONS)
    feature_to_group = {
        feature: group_index + 1
        for group_index, group in enumerate(group_names)
        for feature in GROUP_DEFINITIONS[group]
    }
    if set(feature_to_group) != set(FEATURES):
        raise ValueError("Group definitions must cover every analysis feature")
    groups = np.asarray([feature_to_group[feature] for feature in FEATURES])
    preprocessor = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    x = preprocessor.transform(panel)
    y = panel["log_gmv_next_month"].to_numpy(float)
    y = (y - y.mean()) / y.std()
    sampler = make_sampler(x, groups=groups)
    first = sample_knockoff(sampler, SEED + 24000)
    alpha = choose_alpha(x, first, y)
    runs = []
    for repetition in range(config.group_repetitions):
        xk = (
            first
            if repetition == 0
            else sample_knockoff(sampler, SEED + 24000 + repetition)
        )
        runs.append(group_w(x, xk, y, groups, alpha))
    matrix = np.asarray(runs)
    pd.DataFrame(matrix, columns=group_names).to_csv(
        RESULTS / "group_knockoff_w.csv", index_label="repetition"
    )
    result = summarize_w(
        matrix,
        group_names,
        GROUP_LABELS_ZH,
        "business_group_knockoff",
    )
    result.to_csv(RESULTS / "group_knockoff_selection.csv", index=False)
    membership_rows = [
        {
            "group": group,
            "group_label_zh": GROUP_LABELS_ZH[group],
            "feature": feature,
            "feature_label_zh": FEATURE_LABELS_ZH[feature],
        }
        for group, features in GROUP_DEFINITIONS.items()
        for feature in features
    ]
    pd.DataFrame(membership_rows).to_csv(
        RESULTS / "group_membership.csv", index=False
    )
    return result, {
        "n": len(x),
        "groups": len(group_names),
        "repetitions": config.group_repetitions,
        "lasso_alpha": alpha,
    }


def adversarial_generator_loss(
    x: torch.Tensor,
    xk: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    joint = torch.cat([x, xk], dim=1)
    swapped_x = mask * xk + (1 - mask) * x
    swapped_xk = mask * x + (1 - mask) * xk
    adversarial_swap = mixed_rbf_mmd(
        joint, torch.cat([swapped_x, swapped_xk], dim=1)
    )
    cov_x = covariance(x)
    cov_xk = covariance(xk)
    centered_x = x - x.mean(0)
    centered_xk = xk - xk.mean(0)
    cross = centered_x.T @ centered_xk / max(x.shape[0] - 1, 1)
    paired = torch.diag(cross) / torch.sqrt(
        torch.diag(cov_x).clamp_min(1e-6)
        * torch.diag(cov_xk).clamp_min(1e-6)
    )
    marginal = mixed_rbf_mmd(x, xk)
    covariance_loss = ((cov_xk - cov_x) ** 2).mean()
    symmetry = ((cross - cross.T) ** 2).mean()
    reconstructability = ((paired - 0.50) ** 2).mean()
    loss = (
        12.0 * adversarial_swap
        + 1.5 * marginal
        + 3.0 * covariance_loss
        + 1.5 * symmetry
        + 5.0 * reconstructability
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "worst_swap_mmd": float(adversarial_swap.detach().cpu()),
        "marginal_mmd": float(marginal.detach().cpu()),
        "covariance": float(covariance_loss.detach().cpu()),
        "symmetry": float(symmetry.detach().cpu()),
        "reconstructability": float(reconstructability.detach().cpu()),
        "mask_fraction": float(mask.mean().detach().cpu()),
    }


def train_adversarial_generator(
    x: np.ndarray, config: V4Config, device: torch.device
) -> tuple[DeepKnockoffGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    train_indices, validation_indices = train_test_split(
        np.arange(len(x)), test_size=0.20, random_state=SEED
    )
    x_train = x[train_indices].astype(np.float32)
    x_validation = x[validation_indices].astype(np.float32)
    model = DeepKnockoffGenerator(x.shape[1]).to(device)
    generator_optimizer = torch.optim.AdamW(
        model.parameters(), lr=7e-4, weight_decay=1e-5
    )
    adversary_logits = nn.Parameter(torch.zeros(x.shape[1], device=device))
    adversary_optimizer = torch.optim.Adam([adversary_logits], lr=2e-2)
    best_state = None
    best_score = math.inf
    patience = 0
    history = []
    rng = np.random.default_rng(SEED + 25000)
    for epoch in range(config.generator_epochs):
        model.train()
        order = rng.permutation(len(x_train))
        parts_by_key: dict[str, list[float]] = {}
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            if len(indices) < 32:
                continue
            batch = torch.from_numpy(x_train[indices]).to(device)
            noise = torch.randn_like(batch)
            with torch.no_grad():
                detached_xk = model(batch, noise)
            for _ in range(2):
                temperature = max(0.35, 1.5 * (0.97**epoch))
                uniform = torch.rand_like(adversary_logits).clamp(1e-6, 1 - 1e-6)
                logistic = torch.log(uniform) - torch.log1p(-uniform)
                soft_mask = torch.sigmoid(
                    (adversary_logits + logistic) / temperature
                ).unsqueeze(0)
                joint = torch.cat([batch, detached_xk], dim=1)
                swapped = torch.cat(
                    [
                        soft_mask * detached_xk + (1 - soft_mask) * batch,
                        soft_mask * batch + (1 - soft_mask) * detached_xk,
                    ],
                    dim=1,
                )
                adversary_objective = mixed_rbf_mmd(joint, swapped)
                size_penalty = 0.05 * (soft_mask.mean() - 0.50) ** 2
                adversary_loss = -adversary_objective + size_penalty
                adversary_optimizer.zero_grad(set_to_none=True)
                adversary_loss.backward()
                adversary_optimizer.step()

            hard_prob = torch.sigmoid(adversary_logits).detach()
            hard_mask = (
                (hard_prob >= 0.5).float()
                + hard_prob
                - hard_prob.detach()
            ).unsqueeze(0)
            xk = model(batch, torch.randn_like(batch))
            loss, parts = adversarial_generator_loss(batch, xk, hard_mask)
            generator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            generator_optimizer.step()
            for key, value in parts.items():
                parts_by_key.setdefault(key, []).append(value)

        if epoch % 2 == 0 or epoch == config.generator_epochs - 1:
            raw = generate_torch(
                model, x_validation, device, SEED + 26000 + epoch
            )
            score = quick_generator_score(x_validation, raw)
            validation_score = (
                score["mean_marginal_ks"]
                + score["covariance_relative_error"]
                + 0.5 * score["cross_covariance_asymmetry"]
                + abs(score["mean_real_knockoff_correlation"] - 0.50)
            )
            row = {
                "epoch": epoch + 1,
                **{
                    f"train_{key}": float(np.mean(value))
                    for key, value in parts_by_key.items()
                },
                **{f"validation_{key}": value for key, value in score.items()},
                "validation_score": validation_score,
                "adversary_mean_probability": float(
                    torch.sigmoid(adversary_logits).mean().detach().cpu()
                ),
            }
            history.append(row)
            logging.info(
                "Adversarial generator epoch=%s score=%.4f ks=%.4f cov=%.4f",
                epoch + 1,
                validation_score,
                score["mean_marginal_ks"],
                score["covariance_relative_error"],
            )
            if validation_score < best_score - 1e-4:
                best_score = validation_score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= config.generator_patience:
                    break
    if best_state is None:
        raise RuntimeError("Adversarial generator failed to create a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    torch.save(
        {
            "state_dict": best_state,
            "features": FEATURES,
            "seed": SEED,
            "best_validation_score": best_score,
            "config": asdict(config),
            "adversary_probabilities": torch.sigmoid(adversary_logits)
            .detach()
            .cpu(),
        },
        MODELS / "v4_adversarial_knockoff_generator.pt",
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        RESULTS / "v4_adversarial_generator_history.csv", index=False
    )
    return model, history_frame, train_indices, validation_indices


class DeepKernel(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 24),
            nn.LayerNorm(24),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(values))


def deep_kernel_mmd_test(
    original: np.ndarray,
    swapped: np.ndarray,
    seed: int,
    config: V4Config,
    device: torch.device,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    maximum = min(len(original), 2400)
    indices = rng.choice(len(original), maximum, replace=False)
    original = original[indices].astype(np.float32)
    swapped = swapped[indices].astype(np.float32)
    train_indices, test_indices = train_test_split(
        np.arange(maximum), test_size=0.45, random_state=seed
    )
    model = DeepKernel(original.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    for epoch in range(config.deep_kernel_epochs):
        order = rng.permutation(train_indices)
        for start in range(0, len(order), 256):
            batch_indices = order[start : start + 256]
            if len(batch_indices) < 32:
                continue
            first = torch.from_numpy(original[batch_indices]).to(device)
            second = torch.from_numpy(swapped[batch_indices]).to(device)
            embedded_first = model(first)
            embedded_second = model(second)
            mmd = mixed_rbf_mmd(embedded_first, embedded_second)
            pooled = torch.cat([embedded_first, embedded_second], dim=0)
            variance_floor = torch.relu(0.20 - pooled.var(dim=0).mean())
            loss = -mmd + 0.5 * variance_floor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        embedded_original = (
            model(torch.from_numpy(original[test_indices]).to(device))
            .cpu()
            .numpy()
        )
        embedded_swapped = (
            model(torch.from_numpy(swapped[test_indices]).to(device))
            .cpu()
            .numpy()
        )
    return rbf_mmd_permutation(
        embedded_original,
        embedded_swapped,
        seed + 1,
        config.diagnostic_permutations,
        maximum_per_group=900,
    )


def diagnose_generator(
    real_values: np.ndarray,
    knockoff: np.ndarray,
    method: str,
    config: V4Config,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    base = quick_generator_score(real_values, knockoff)
    rng = np.random.default_rng(SEED + 27000)
    joint = np.hstack([real_values, knockoff])
    rows = []
    for index, ratio in enumerate((0.10, 0.30, 0.50, 0.70, 1.00)):
        count = max(1, int(round(ratio * real_values.shape[1])))
        columns = rng.choice(real_values.shape[1], count, replace=False)
        swapped_real = real_values.copy()
        swapped_knockoff = knockoff.copy()
        swapped_real[:, columns] = knockoff[:, columns]
        swapped_knockoff[:, columns] = real_values[:, columns]
        swapped_joint = np.hstack([swapped_real, swapped_knockoff])
        rbf, rbf_p = rbf_mmd_permutation(
            joint,
            swapped_joint,
            SEED + 27100 + index,
            config.diagnostic_permutations,
        )
        deep, deep_p = deep_kernel_mmd_test(
            joint,
            swapped_joint,
            SEED + 27200 + index,
            config,
            device,
        )
        auc, auc_p = swap_classifier_auc(
            joint, swapped_joint, SEED + 27300 + index
        )
        rows.append(
            {
                "method": method,
                "swap_ratio": ratio,
                "swapped_features": count,
                "rbf_mmd": rbf,
                "rbf_mmd_p": rbf_p,
                "deep_kernel_mmd": deep,
                "deep_kernel_mmd_p": deep_p,
                "classifier_auc": auc,
                "classifier_p": auc_p,
                "kernel_permutations": config.diagnostic_permutations,
                "kernel_minimum_attainable_p": 1
                / (config.diagnostic_permutations + 1),
                "classifier_permutations": 99,
                "classifier_minimum_attainable_p": 0.01,
            }
        )
    frame = pd.DataFrame(rows)
    base.update(
        {
            "method": method,
            "deep_kernel_mmd_mean": float(frame["deep_kernel_mmd"].mean()),
            "deep_kernel_rejection_rate_5pct": float(
                (frame["deep_kernel_mmd_p"] < 0.05).mean()
            ),
            "classifier_auc_mean": float(frame["classifier_auc"].mean()),
            "classifier_rejection_rate_5pct": float(
                (frame["classifier_p"] < 0.05).mean()
            ),
        }
    )
    return base, frame


class GRIP2StyleMLP(PairwiseCompetitiveMLP):
    """Pairwise MLP with trajectory-integrated two-regularizer importance."""


def train_grip2_style(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    seed: int,
    config: V4Config,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    p = x.shape[1]
    pair_swap = rng.random(p) < 0.5
    first = x.copy()
    second = xk.copy()
    first[:, pair_swap], second[:, pair_swap] = (
        second[:, pair_swap].copy(),
        first[:, pair_swap].copy(),
    )
    train_indices, validation_indices = train_test_split(
        np.arange(len(x)), test_size=0.20, random_state=seed
    )
    y_mean = y[train_indices].mean()
    y_std = y[train_indices].std() or 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    model = GRIP2StyleMLP(p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    mse = nn.MSELoss()
    trajectory_real = []
    trajectory_knockoff = []
    best_validation = math.inf
    best_state = None
    patience = 0
    lambda_l1_grid = (1e-5, 5e-5, 2e-4, 8e-4)
    lambda_group_grid = (1e-5, 1e-4, 5e-4, 2e-3)
    for epoch in range(config.grip_epochs):
        model.train()
        lambda_l1 = lambda_l1_grid[epoch % len(lambda_l1_grid)]
        lambda_group = lambda_group_grid[
            (epoch // len(lambda_l1_grid)) % len(lambda_group_grid)
        ]
        order = rng.permutation(train_indices)
        for start in range(0, len(order), 512):
            indices = order[start : start + 512]
            xb = torch.from_numpy(first[indices]).float().to(device)
            xkb = torch.from_numpy(second[indices]).float().to(device)
            yb = torch.from_numpy(y_scaled[indices]).to(device)
            prediction = model(xb, xkb)
            l1 = (
                model.real_weight.abs().mean()
                + model.knockoff_weight.abs().mean()
            )
            group = torch.sqrt(
                model.real_weight.square()
                + model.knockoff_weight.square()
                + 1e-8
            ).mean()
            loss = mse(prediction, yb) + lambda_l1 * l1 + lambda_group * group
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            real, knockoff = model.importance()
            if epoch >= 5:
                trajectory_real.append(real)
                trajectory_knockoff.append(knockoff)
            validation_prediction = model(
                torch.from_numpy(first[validation_indices]).float().to(device),
                torch.from_numpy(second[validation_indices]).float().to(device),
            )
            validation = float(
                mse(
                    validation_prediction,
                    torch.from_numpy(y_scaled[validation_indices]).to(device),
                )
                .cpu()
                .numpy()
            )
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 16 and epoch >= 35:
            break
    if not trajectory_real or best_state is None:
        raise RuntimeError("GRIP2-style model failed")
    integrated_real = np.mean(np.asarray(trajectory_real), axis=0)
    integrated_knockoff = np.mean(np.asarray(trajectory_knockoff), axis=0)
    w = integrated_real - integrated_knockoff
    w[pair_swap] *= -1
    return w, {
        "seed": seed,
        "epochs": epoch + 1,
        "trajectory_points": len(trajectory_real),
        "best_validation_mse": best_validation,
        "lambda_l1_grid": lambda_l1_grid,
        "lambda_group_grid": lambda_group_grid,
    }


def grip_antisymmetry_check(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    config: V4Config,
) -> dict:
    rng = np.random.default_rng(SEED + 30000)
    indices = rng.choice(len(x), min(2400, len(x)), replace=False)
    small_x = x[indices].copy()
    small_xk = xk[indices].copy()
    small_y = y[indices].copy()
    short_config = V4Config(**asdict(config))
    short_config.grip_epochs = 35
    base, _ = train_grip2_style(
        small_x,
        small_xk,
        small_y,
        SEED + 30100,
        short_config,
        torch.device("cpu"),
    )
    swapped_x = small_x.copy()
    swapped_xk = small_xk.copy()
    swapped_x[:, 0], swapped_xk[:, 0] = (
        small_xk[:, 0].copy(),
        small_x[:, 0].copy(),
    )
    swapped, _ = train_grip2_style(
        swapped_x,
        swapped_xk,
        small_y,
        SEED + 30100,
        short_config,
        torch.device("cpu"),
    )
    expected = base.copy()
    expected[0] *= -1
    error = np.abs(swapped - expected)
    return {
        "feature": FEATURES[0],
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "passed_tolerance_1e_5": bool(error.max() <= 1e-5),
    }


def run_grip2_style(
    panel: pd.DataFrame, config: V4Config, device: torch.device
) -> tuple[pd.DataFrame, dict]:
    preprocessor = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    x = preprocessor.transform(panel).astype(np.float32)
    y = panel["log_gmv_next_month"].to_numpy(np.float32)
    sampler = make_sampler(x)
    runs = []
    metadata = []
    first_xk = None
    for repetition in range(config.grip_repetitions):
        xk = sample_knockoff(
            sampler, SEED + 31000 + repetition
        ).astype(np.float32)
        if first_xk is None:
            first_xk = xk.copy()
        w, run_metadata = train_grip2_style(
            x,
            xk,
            y,
            SEED + 32000 + repetition,
            config,
            device,
        )
        runs.append(w)
        metadata.append(run_metadata)
        logging.info("GRIP2-style repetition %s completed", repetition + 1)
    matrix = np.asarray(runs)
    pd.DataFrame(matrix, columns=FEATURES).to_csv(
        RESULTS / "grip2_style_w.csv", index_label="repetition"
    )
    selection = summarize_w(
        matrix, FEATURES, FEATURE_LABELS_ZH, "grip2_style_pairwise_mlp"
    )
    selection.to_csv(RESULTS / "grip2_style_selection.csv", index=False)
    antisymmetry = grip_antisymmetry_check(x, first_xk, y, config)
    output_metadata = {
        "runs": metadata,
        "antisymmetry": antisymmetry,
    }
    (RESULTS / "grip2_style_metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selection, output_metadata


def make_figures(
    clustered: pd.DataFrame,
    grouped: pd.DataFrame,
    diagnostics: pd.DataFrame,
    grip: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    clustered_q = clustered.loc[clustered["q"].eq(0.20)]
    pivot = clustered_q.pivot(
        index="label_zh", columns="method", values="selection_frequency"
    ).fillna(0)
    pivot.max(axis=1).nlargest(14).sort_values().plot.barh(ax=axes[0])
    axes[0].set_title("聚类分层Knockoff稳定频率")
    axes[0].set_xlabel("单轮入选频率")
    group_q = grouped.loc[grouped["q"].eq(0.20)].sort_values(
        "selection_frequency"
    )
    axes[1].barh(
        group_q["label_zh"],
        group_q["selection_frequency"],
        color="#287271",
    )
    axes[1].set_title("业务组Knockoff稳定频率")
    axes[1].set_xlabel("单轮入选频率")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v4_cluster_group.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, frame in diagnostics.groupby("method"):
        axes[0].plot(
            frame["swap_ratio"],
            frame["deep_kernel_mmd"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            frame["swap_ratio"],
            frame["classifier_auc"],
            marker="o",
            label=method,
        )
    axes[0].set_title("深度核MMD")
    axes[0].set_xlabel("swap比例")
    axes[1].set_title("swap分类器AUC")
    axes[1].set_xlabel("swap比例")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v4_exchangeability.png", dpi=220)
    plt.close(fig)

    grip_q = grip.loc[grip["q"].eq(0.20)].nlargest(15, "mean_w").sort_values(
        "mean_w"
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(grip_q["label_zh"], grip_q["mean_w"], color="#C44536")
    ax.set_title("GRIP2式轨迹聚合重要性")
    ax.set_xlabel("平均W")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_v4_grip_importance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    configure_logging()
    set_seed(SEED)
    config = V4Config()
    device = get_device()
    panel = pd.read_csv(
        V1_DIR / "data_processed" / "seller_month_panel.csv",
        parse_dates=["month"],
    )
    logging.info("V4 analysis started device=%s", device)

    clustered, clustered_metadata = run_clustered_knockoffs(panel, config)
    grouped, group_metadata = run_group_knockoff(panel, config)

    preprocessor = RobustPreprocessor(FEATURES).fit(panel, copula=False)
    x_base = preprocessor.transform(panel).astype(np.float32)
    generator, history, train_indices, validation_indices = (
        train_adversarial_generator(x_base, config, device)
    )
    raw_training = generate_torch(
        generator, x_base[train_indices], device, SEED + 33000
    )
    calibrator = EmpiricalMarginalCalibrator().fit(
        raw_training, x_base[train_indices]
    )
    raw_all = generate_torch(generator, x_base, device, SEED + 33100)
    calibrated_all = calibrator.transform(raw_all)
    np.savez_compressed(
        MODELS / "v4_train_only_marginal_calibrator.npz",
        **{
            f"raw_{index}": value
            for index, value in enumerate(calibrator.raw_sorted)
        },
        **{
            f"target_{index}": value
            for index, value in enumerate(calibrator.target_sorted)
        },
    )

    copula_preprocessor = RobustPreprocessor(FEATURES).fit(panel, copula=True)
    x_copula = copula_preprocessor.transform(panel)
    copula_xk = sample_knockoff(make_sampler(x_copula), SEED + 33200)

    v2_checkpoint = torch.load(
        V2_DIR / "models" / "deep_knockoff_generator.pt",
        map_location="cpu",
        weights_only=True,
    )
    v2_generator = DeepKnockoffGenerator(len(FEATURES))
    v2_generator.load_state_dict(v2_checkpoint["state_dict"])
    v2_generator.to(device).eval()
    v2_raw_training = generate_torch(
        v2_generator, x_base[train_indices], device, SEED + 33300
    )
    v2_calibrator = EmpiricalMarginalCalibrator().fit(
        v2_raw_training, x_base[train_indices]
    )
    v2_raw_all = generate_torch(
        v2_generator, x_base, device, SEED + 33400
    )
    v2_calibrated = v2_calibrator.transform(v2_raw_all)

    diagnostic_summaries = []
    diagnostic_frames = []
    for method, real, knockoff in [
        (
            "copula_mvr",
            x_copula[validation_indices],
            copula_xk[validation_indices],
        ),
        (
            "v2_deep_train_calibrated",
            x_base[validation_indices],
            v2_calibrated[validation_indices],
        ),
        (
            "v4_worst_swap_train_calibrated",
            x_base[validation_indices],
            calibrated_all[validation_indices],
        ),
    ]:
        summary, frame = diagnose_generator(
            real, knockoff, method, config, device
        )
        diagnostic_summaries.append(summary)
        diagnostic_frames.append(frame)
        logging.info("V4 diagnostics %s: %s", method, summary)
    diagnostic_summary = pd.DataFrame(diagnostic_summaries)
    diagnostic_detail = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostic_summary.to_csv(
        RESULTS / "v4_generator_diagnostic_summary.csv", index=False
    )
    diagnostic_detail.to_csv(
        RESULTS / "v4_exchangeability_diagnostics.csv", index=False
    )

    alpha = float(
        json.loads(
            (V1_DIR / "results" / "knockoff_metadata.json").read_text()
        )[0]["lasso_alpha"]
    )
    deep_runs = []
    for repetition in range(config.generator_repetitions):
        raw_train = generate_torch(
            generator,
            x_base[train_indices],
            device,
            SEED + 34000 + repetition,
        )
        repetition_calibrator = EmpiricalMarginalCalibrator().fit(
            raw_train, x_base[train_indices]
        )
        raw = generate_torch(
            generator, x_base, device, SEED + 35000 + repetition
        )
        xk = repetition_calibrator.transform(raw)
        deep_runs.append(lasso_w(x_base, xk, panel["log_gmv_next_month"], alpha))
    deep_matrix = np.asarray(deep_runs)
    pd.DataFrame(deep_matrix, columns=FEATURES).to_csv(
        RESULTS / "v4_adversarial_generator_w.csv", index_label="repetition"
    )
    deep_selection = summarize_w(
        deep_matrix,
        FEATURES,
        FEATURE_LABELS_ZH,
        "v4_worst_swap_generator_lasso",
    )
    deep_selection.to_csv(
        RESULTS / "v4_adversarial_generator_selection.csv", index=False
    )

    grip, grip_metadata = run_grip2_style(panel, config, device)
    make_figures(clustered, grouped, diagnostic_detail, grip)

    diagnostic_index = diagnostic_summary.set_index("method")
    v4_row = diagnostic_index.loc["v4_worst_swap_train_calibrated"]
    v4_deep_valid = bool(
        v4_row["mean_marginal_ks"] <= 0.05
        and v4_row["covariance_relative_error"] <= 0.20
        and abs(v4_row["mean_real_knockoff_correlation"]) <= 0.85
        and v4_row["deep_kernel_rejection_rate_5pct"] <= 0.40
        and v4_row["classifier_auc_mean"] <= 0.75
    )
    metadata = {
        "config": asdict(config),
        "seed": SEED,
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "panel_rows": len(panel),
        "panel_sellers": panel["seller_id"].nunique(),
        "clustered": clustered_metadata,
        "group": group_metadata,
        "v4_generator_passed_prespecified_diagnostics": v4_deep_valid,
        "v4_generator_diagnostic_rule": {
            "mean_marginal_ks_max": 0.05,
            "covariance_relative_error_max": 0.20,
            "mean_real_knockoff_correlation_max": 0.85,
            "deep_kernel_rejection_rate_max": 0.40,
            "classifier_auc_mean_max": 0.75,
        },
        "marginal_calibration_fit_scope": "generator_training_indices_only",
        "diagnostic_permutations": config.diagnostic_permutations,
        "minimum_attainable_p": 1 / (config.diagnostic_permutations + 1),
        "classifier_permutations": 99,
        "classifier_minimum_attainable_p": 0.01,
        "grip2_style_antisymmetry": grip_metadata["antisymmetry"],
        "panel_sha256": json.loads(
            (V1_DIR / "results" / "data_audit.json").read_text()
        )["panel_sha256"],
    }
    (RESULTS / "v4_reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("V4 analysis completed: %s", metadata)


if __name__ == "__main__":
    main()
