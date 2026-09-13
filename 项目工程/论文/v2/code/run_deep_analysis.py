#!/usr/bin/env python3
"""Run the V2 deep-learning extensions on the verified V1 panel."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = HERE.parents[3]
V1_DIR = PROJECT_DIR / "论文" / "v1"
sys.path.insert(0, str(V1_DIR / "code"))

from build_panel import FEATURES, FEATURE_LABELS_ZH  # noqa: E402
from run_analysis import (  # noqa: E402
    RobustPreprocessor,
    ebh,
    make_sampler,
    metric_row,
    sample_knockoff,
    threshold_knockoff_plus,
)


RESULTS_DIR = V2_DIR / "results"
FIGURES_DIR = V2_DIR / "figures"
LOG_DIR = V2_DIR / "logs"
MODEL_DIR = V2_DIR / "models"
SEED = 20260823

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "STHeiti",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class RunConfig:
    mode: str
    generator_epochs: int
    generator_patience: int
    generator_repetitions: int
    paired_mlp_repetitions: int
    paired_mlp_epochs: int
    predictor_epochs: int
    diagnostic_permutations: int


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(output_dir / "v2_deep_analysis.log", mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeepKnockoffGenerator(nn.Module):
    """Noise-conditional generator trained with swap and moment objectives."""

    def __init__(self, p: int, hidden: int = 192, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 2 * p
        for layer_index in range(depth):
            output_dim = hidden
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden, p))
        self.network = nn.Sequential(*layers)
        self.output_scale = nn.Parameter(torch.ones(p))

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        raw = self.network(torch.cat([x, noise], dim=1))
        return 4.5 * torch.tanh(raw * self.output_scale / 4.5)


def covariance(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(x.shape[0] - 1, 1)


def mixed_rbf_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    dimension = x.shape[1]

    def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True).T
        return (aa + bb - 2 * a @ b.T).clamp_min(0) / dimension

    dxx = squared_distance(x, x)
    dyy = squared_distance(y, y)
    dxy = squared_distance(x, y)
    value = torch.zeros((), device=x.device)
    for scale in scales:
        denominator = 2 * scale * scale
        value = value + (
            torch.exp(-dxx / denominator).mean()
            + torch.exp(-dyy / denominator).mean()
            - 2 * torch.exp(-dxy / denominator).mean()
        )
    return value / len(scales)


def generator_batch_loss(
    x: torch.Tensor,
    xk: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    p = x.shape[1]
    joint = torch.cat([x, xk], dim=1)
    swap_losses = []
    for ratio in (0.20, 0.50, 0.80):
        swap_mask = torch.rand(1, p, device=x.device) < ratio
        swapped_x = torch.where(swap_mask, xk, x)
        swapped_xk = torch.where(swap_mask, x, xk)
        swapped_joint = torch.cat([swapped_x, swapped_xk], dim=1)
        swap_losses.append(mixed_rbf_mmd(joint, swapped_joint))
    swap_loss = torch.stack(swap_losses).mean()
    marginal_loss = mixed_rbf_mmd(x, xk)

    cov_x = covariance(x)
    cov_xk = covariance(xk)
    cross = (x - x.mean(0)).T @ (xk - xk.mean(0)) / max(x.shape[0] - 1, 1)
    target_cross = cov_x - torch.diag(0.5 * torch.diag(cov_x))
    covariance_loss = ((cov_xk - cov_x) ** 2).mean()
    cross_loss = ((cross - target_cross) ** 2).mean()
    symmetry_loss = ((cross - cross.T) ** 2).mean()
    paired_correlation = torch.diag(cross) / torch.sqrt(
        torch.diag(cov_x).clamp_min(1e-6)
        * torch.diag(cov_xk).clamp_min(1e-6)
    )
    reconstructability_loss = ((paired_correlation - 0.50) ** 2).mean()
    mean_loss = ((xk.mean(0) - x.mean(0)) ** 2).mean()
    variance_loss = ((xk.var(0, unbiased=False) - x.var(0, unbiased=False)) ** 2).mean()

    loss = (
        8.0 * swap_loss
        + 1.5 * marginal_loss
        + 3.0 * covariance_loss
        + 2.0 * cross_loss
        + 1.0 * symmetry_loss
        + 5.0 * reconstructability_loss
        + 0.5 * mean_loss
        + 0.5 * variance_loss
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "swap_mmd": float(swap_loss.detach().cpu()),
        "marginal_mmd": float(marginal_loss.detach().cpu()),
        "covariance": float(covariance_loss.detach().cpu()),
        "cross": float(cross_loss.detach().cpu()),
        "symmetry": float(symmetry_loss.detach().cpu()),
        "reconstructability": float(reconstructability_loss.detach().cpu()),
    }
    return loss, parts


def generate_torch(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int = 1024,
) -> np.ndarray:
    set_seed(seed)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            noise = torch.randn_like(batch)
            outputs.append(model(batch, noise).cpu().numpy())
    return np.vstack(outputs)


def calibrate_marginals(xk: np.ndarray, reference: np.ndarray) -> np.ndarray:
    calibrated = np.empty_like(xk)
    n = len(xk)
    quantiles = (np.arange(n) + 0.5) / n
    for column in range(xk.shape[1]):
        order = np.argsort(xk[:, column], kind="mergesort")
        values = np.quantile(reference[:, column], quantiles, method="linear")
        calibrated[order, column] = values
    return calibrated


def quick_generator_score(x: np.ndarray, xk: np.ndarray) -> dict[str, float]:
    ks = [
        stats.ks_2samp(x[:, j], xk[:, j]).statistic
        for j in range(x.shape[1])
    ]
    covariance_error = np.linalg.norm(
        np.cov(x, rowvar=False) - np.cov(xk, rowvar=False), ord="fro"
    ) / max(np.linalg.norm(np.cov(x, rowvar=False), ord="fro"), 1e-12)
    cross = np.cov(x.T, xk.T)[: x.shape[1], x.shape[1] :]
    cross_asymmetry = np.linalg.norm(cross - cross.T, ord="fro") / max(
        np.linalg.norm(cross, ord="fro"), 1e-12
    )
    paired_correlations = []
    for j in range(x.shape[1]):
        if np.std(x[:, j]) > 1e-8 and np.std(xk[:, j]) > 1e-8:
            paired_correlations.append(np.corrcoef(x[:, j], xk[:, j])[0, 1])
    return {
        "mean_marginal_ks": float(np.mean(ks)),
        "max_marginal_ks": float(np.max(ks)),
        "covariance_relative_error": float(covariance_error),
        "cross_covariance_asymmetry": float(cross_asymmetry),
        "mean_real_knockoff_correlation": float(np.mean(paired_correlations)),
    }


def train_generator(
    x: np.ndarray,
    config: RunConfig,
    device: torch.device,
    model_dir: Path,
) -> tuple[DeepKnockoffGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    train_indices, validation_indices = train_test_split(
        np.arange(len(x)), test_size=0.20, random_state=SEED
    )
    x_train = x[train_indices].astype(np.float32)
    x_validation = x[validation_indices].astype(np.float32)
    model = DeepKnockoffGenerator(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.65, patience=5, min_lr=2e-5
    )

    batch_size = 256 if config.mode == "full" else 192
    history = []
    best_state = None
    best_score = math.inf
    patience_counter = 0
    start_time = time.time()
    for epoch in range(config.generator_epochs):
        model.train()
        permutation = np.random.default_rng(SEED + epoch).permutation(len(x_train))
        epoch_parts: dict[str, list[float]] = {}
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            if len(indices) < 32:
                continue
            batch = torch.from_numpy(x_train[indices]).to(device)
            noise = torch.randn_like(batch)
            xk = model(batch, noise)
            loss, parts = generator_batch_loss(batch, xk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in parts.items():
                epoch_parts.setdefault(key, []).append(value)

        if epoch % 2 == 0 or epoch == config.generator_epochs - 1:
            raw_validation = generate_torch(
                model, x_validation, device, SEED + 90000 + epoch
            )
            score_parts = quick_generator_score(x_validation, raw_validation)
            validation_score = (
                score_parts["mean_marginal_ks"]
                + score_parts["covariance_relative_error"]
                + 0.5 * score_parts["cross_covariance_asymmetry"]
                + abs(
                    score_parts["mean_real_knockoff_correlation"] - 0.50
                )
            )
            scheduler.step(validation_score)
            row = {
                "epoch": epoch + 1,
                **{
                    f"train_{key}": float(np.mean(values))
                    for key, values in epoch_parts.items()
                },
                **{f"validation_{key}": value for key, value in score_parts.items()},
                "validation_score": validation_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            logging.info(
                "Generator epoch=%s loss=%.5f val_score=%.5f ks=%.4f cov=%.4f",
                epoch + 1,
                row["train_loss"],
                validation_score,
                score_parts["mean_marginal_ks"],
                score_parts["covariance_relative_error"],
            )
            if validation_score < best_score - 1e-4:
                best_score = validation_score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.generator_patience:
                    logging.info("Generator early stopping at epoch %s", epoch + 1)
                    break

    if best_state is None:
        raise RuntimeError("Deep generator did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "features": FEATURES,
            "seed": SEED,
            "best_validation_score": best_score,
            "config": asdict(config),
        },
        model_dir / "deep_knockoff_generator.pt",
    )
    logging.info(
        "Generator training completed in %.1f seconds; best score %.5f",
        time.time() - start_time,
        best_score,
    )
    return model, pd.DataFrame(history), train_indices, validation_indices


def load_generator(
    x: np.ndarray,
    device: torch.device,
    model_dir: Path,
    result_dir: Path,
) -> tuple[DeepKnockoffGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    checkpoint_path = model_dir / "deep_knockoff_generator.pt"
    history_path = result_dir / "deep_generator_training_history.csv"
    if not checkpoint_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            "Generator checkpoint and training history are required for resume mode"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["features"] != FEATURES:
        raise ValueError("Generator checkpoint features do not match V1 features")
    model = DeepKnockoffGenerator(x.shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    train_indices, validation_indices = train_test_split(
        np.arange(len(x)), test_size=0.20, random_state=SEED
    )
    logging.info(
        "Reused generator checkpoint with best validation score %.5f",
        checkpoint["best_validation_score"],
    )
    return (
        model,
        pd.read_csv(history_path),
        train_indices,
        validation_indices,
    )


def empirical_quantile_inverse(
    gaussian_values: np.ndarray, target_values: np.ndarray
) -> np.ndarray:
    output = np.empty_like(gaussian_values)
    for j in range(gaussian_values.shape[1]):
        probabilities = np.clip(stats.norm.cdf(gaussian_values[:, j]), 1e-6, 1 - 1e-6)
        output[:, j] = np.quantile(
            target_values[:, j], probabilities, method="linear"
        )
    return output


def rbf_mmd_permutation(
    a: np.ndarray,
    b: np.ndarray,
    seed: int,
    permutations: int,
    maximum_per_group: int = 650,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b), maximum_per_group)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    pooled = np.clip(
        np.vstack([a, b]).astype(np.float64),
        -20,
        20,
    )
    if not np.isfinite(pooled).all():
        raise ValueError("Non-finite values supplied to the MMD diagnostic")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        norms = np.sum(pooled * pooled, axis=1, keepdims=True)
        distances = np.maximum(norms + norms.T - 2 * pooled @ pooled.T, 0)
    distances = np.nan_to_num(distances, nan=0.0, posinf=1600.0, neginf=0.0)
    off_diagonal = distances[np.triu_indices_from(distances, k=1)]
    positive = off_diagonal[off_diagonal > 0]
    bandwidth = np.median(positive) if len(positive) else 1.0
    kernel = np.exp(-distances / max(2 * bandwidth, 1e-12))

    def statistic(index_a: np.ndarray, index_b: np.ndarray) -> float:
        kaa = kernel[np.ix_(index_a, index_a)]
        kbb = kernel[np.ix_(index_b, index_b)]
        kab = kernel[np.ix_(index_a, index_b)]
        return float(kaa.mean() + kbb.mean() - 2 * kab.mean())

    first = np.arange(n)
    second = np.arange(n, 2 * n)
    observed = statistic(first, second)
    null_values = []
    for _ in range(permutations):
        order = rng.permutation(2 * n)
        null_values.append(statistic(order[:n], order[n:]))
    p_value = (1 + np.sum(np.asarray(null_values) >= observed)) / (
        permutations + 1
    )
    return observed, float(p_value)


def swap_classifier_auc(
    original: np.ndarray,
    swapped: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    row_train, row_test = train_test_split(
        np.arange(len(original)), test_size=0.35, random_state=seed
    )
    x_train = np.vstack([original[row_train], swapped[row_train]])
    y_train = np.concatenate(
        [np.zeros(len(row_train)), np.ones(len(row_train))]
    )
    x_test = np.vstack([original[row_test], swapped[row_test]])
    y_test = np.concatenate([np.zeros(len(row_test)), np.ones(len(row_test))])
    model = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=12,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probability)
    rng = np.random.default_rng(seed + 1)
    null_auc = [
        roc_auc_score(rng.permutation(y_test), probability)
        for _ in range(99)
    ]
    p_value = (1 + np.sum(np.asarray(null_auc) >= auc)) / 100
    return float(auc), float(p_value)


def coverage_metrics(x: np.ndarray, xk: np.ndarray) -> dict[str, float]:
    output = {}
    for level in (0.90, 0.95):
        tail = (1 - level) / 2
        lower = np.quantile(xk, tail, axis=0)
        upper = np.quantile(xk, 1 - tail, axis=0)
        per_feature = ((x >= lower) & (x <= upper)).mean(axis=0)
        output[f"coverage_{int(level * 100)}_mean"] = float(per_feature.mean())
        output[f"coverage_{int(level * 100)}_minimum"] = float(
            per_feature.min()
        )
    return output


def full_diagnostics(
    x: np.ndarray,
    xk: np.ndarray,
    method: str,
    config: RunConfig,
) -> tuple[dict, pd.DataFrame]:
    base = quick_generator_score(x, xk)
    base.update(coverage_metrics(x, xk))
    corr_x = np.corrcoef(x, rowvar=False)
    corr_xk = np.corrcoef(xk, rowvar=False)
    base["correlation_relative_error"] = float(
        np.linalg.norm(corr_x - corr_xk, ord="fro")
        / max(np.linalg.norm(corr_x, ord="fro"), 1e-12)
    )
    base["method"] = method

    rng = np.random.default_rng(SEED + 811)
    rows = []
    joint = np.hstack([x, xk])
    for index, ratio in enumerate((0.10, 0.30, 0.50, 0.70, 1.00)):
        count = max(1, int(round(ratio * x.shape[1])))
        columns = rng.choice(x.shape[1], count, replace=False)
        swapped_x = x.copy()
        swapped_xk = xk.copy()
        swapped_x[:, columns] = xk[:, columns]
        swapped_xk[:, columns] = x[:, columns]
        swapped_joint = np.hstack([swapped_x, swapped_xk])
        mmd, mmd_p = rbf_mmd_permutation(
            joint,
            swapped_joint,
            SEED + index,
            config.diagnostic_permutations,
        )
        auc, auc_p = swap_classifier_auc(
            joint,
            swapped_joint,
            SEED + 100 + index,
        )
        rows.append(
            {
                "method": method,
                "swap_ratio": ratio,
                "swapped_features": count,
                "mmd_rbf": mmd,
                "mmd_permutation_p": mmd_p,
                "classifier_auc": auc,
                "classifier_permutation_p": auc_p,
            }
        )
    frame = pd.DataFrame(rows)
    base["swap_mmd_mean"] = float(frame["mmd_rbf"].mean())
    base["swap_classifier_auc_mean"] = float(frame["classifier_auc"].mean())
    base["swap_mmd_rejection_rate_5pct"] = float(
        (frame["mmd_permutation_p"] < 0.05).mean()
    )
    base["swap_classifier_rejection_rate_5pct"] = float(
        (frame["classifier_permutation_p"] < 0.05).mean()
    )
    return base, frame


def lasso_w_fixed_alpha(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    from sklearn.linear_model import Lasso

    model = Lasso(alpha=alpha, max_iter=10000, tol=1e-5, selection="cyclic")
    model.fit(np.hstack([x, xk]), y)
    p = x.shape[1]
    return np.abs(model.coef_[:p]) - np.abs(model.coef_[p:])


def choose_alpha(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
) -> float:
    from sklearn.linear_model import LassoCV

    grid = np.logspace(-4, -0.15, 28)
    model = LassoCV(
        alphas=grid,
        cv=5,
        max_iter=10000,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(np.hstack([x, xk]), y)
    return float(model.alpha_)


def load_v1_lasso_alpha() -> float:
    metadata = json.loads(
        (V1_DIR / "results" / "knockoff_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    primary = next(
        row
        for row in metadata
        if row["generator"] == "copula" and row["stat_model"] == "lasso"
    )
    return float(primary["lasso_alpha"])


def summarize_w(
    w_matrix: np.ndarray,
    method: str,
    q_values: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    rows = []
    p = w_matrix.shape[1]
    for q in q_values:
        alpha_kn = q / 2
        e_runs = np.zeros_like(w_matrix)
        selected_runs = np.zeros_like(w_matrix, dtype=bool)
        thresholds = []
        for repetition, w in enumerate(w_matrix):
            threshold = threshold_knockoff_plus(w, alpha_kn)
            thresholds.append(threshold)
            if np.isfinite(threshold):
                selected = w >= threshold
                denominator = 1 + np.sum(w <= -threshold)
                e_runs[repetition, selected] = p / denominator
                selected_runs[repetition, selected] = True
        mean_e = e_runs.mean(axis=0)
        final = set(ebh(mean_e, q).tolist())
        for j, feature in enumerate(FEATURES):
            rows.append(
                {
                    "method": method,
                    "q": q,
                    "alpha_kn": alpha_kn,
                    "feature": feature,
                    "label_zh": FEATURE_LABELS_ZH[feature],
                    "mean_w": float(w_matrix[:, j].mean()),
                    "positive_w_rate": float((w_matrix[:, j] > 0).mean()),
                    "selection_frequency": float(selected_runs[:, j].mean()),
                    "mean_evalue": float(mean_e[j]),
                    "selected_ebh": j in final,
                    "median_threshold": float(
                        np.median(
                            [value for value in thresholds if np.isfinite(value)]
                        )
                    )
                    if any(np.isfinite(thresholds))
                    else np.inf,
                }
            )
    return pd.DataFrame(rows)


class PairwiseCompetitiveMLP(nn.Module):
    """DeepPINK-style paired input layer with antisymmetric importance."""

    def __init__(self, p: int, hidden: int = 128) -> None:
        super().__init__()
        self.real_weight = nn.Parameter(torch.ones(p))
        self.knockoff_weight = nn.Parameter(torch.ones(p))
        self.first = nn.Linear(p, hidden)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:
        paired = x * self.real_weight + xk * self.knockoff_weight
        return self.network(self.first(paired)).squeeze(1)

    def importance(self) -> tuple[np.ndarray, np.ndarray]:
        downstream = torch.sqrt((self.first.weight**2).sum(dim=0) + 1e-12)
        real = torch.abs(self.real_weight) * downstream
        knockoff = torch.abs(self.knockoff_weight) * downstream
        return real.detach().cpu().numpy(), knockoff.detach().cpu().numpy()


def train_pairwise_mlp(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    p = x.shape[1]
    rng = np.random.default_rng(seed)
    swap_pairs = rng.random(p) < 0.5
    first = x.copy()
    second = xk.copy()
    first[:, swap_pairs], second[:, swap_pairs] = (
        second[:, swap_pairs].copy(),
        first[:, swap_pairs].copy(),
    )
    train_idx, valid_idx = train_test_split(
        np.arange(len(x)), test_size=0.20, random_state=seed
    )
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() or 1.0)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    model = PairwiseCompetitiveMLP(p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    best_state = None
    best_validation = math.inf
    patience = 10
    patience_counter = 0
    batch_size = 512
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            xb = torch.from_numpy(first[indices]).float().to(device)
            xkb = torch.from_numpy(second[indices]).float().to(device)
            yb = torch.from_numpy(y_scaled[indices]).to(device)
            prediction = model(xb, xkb)
            sparsity = (
                torch.abs(model.real_weight).mean()
                + torch.abs(model.knockoff_weight).mean()
            )
            loss = loss_function(prediction, yb) + 2e-4 * sparsity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(first[valid_idx]).float().to(device)
            xkv = torch.from_numpy(second[valid_idx]).float().to(device)
            yv = torch.from_numpy(y_scaled[valid_idx]).to(device)
            validation = float(
                loss_function(model(xv, xkv), yv).detach().cpu()
            )
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is None:
        raise RuntimeError("Pairwise MLP failed to train")
    model.load_state_dict(best_state)
    real, knockoff = model.importance()
    w = real - knockoff
    w[swap_pairs] *= -1
    return w, {
        "seed": seed,
        "epochs": epoch + 1,
        "best_validation_mse": best_validation,
        "swapped_pair_count": int(swap_pairs.sum()),
    }


def paired_mlp_antisymmetry_check(
    x: np.ndarray,
    xk: np.ndarray,
    y: np.ndarray,
    max_epochs: int,
) -> dict:
    sample_size = min(2600, len(x))
    indices = np.random.default_rng(SEED + 155000).choice(
        len(x), sample_size, replace=False
    )
    x_small = x[indices].copy()
    xk_small = xk[indices].copy()
    y_small = y[indices].copy()
    test_feature = 0
    seed = SEED + 156000
    epochs = min(max_epochs, 30)
    base_w, _ = train_pairwise_mlp(
        x_small,
        xk_small,
        y_small,
        seed,
        epochs,
        torch.device("cpu"),
    )
    swapped_x = x_small.copy()
    swapped_xk = xk_small.copy()
    swapped_x[:, test_feature] = xk_small[:, test_feature]
    swapped_xk[:, test_feature] = x_small[:, test_feature]
    swapped_w, _ = train_pairwise_mlp(
        swapped_x,
        swapped_xk,
        y_small,
        seed,
        epochs,
        torch.device("cpu"),
    )
    expected = base_w.copy()
    expected[test_feature] *= -1
    absolute_error = np.abs(swapped_w - expected)
    return {
        "sample_size": sample_size,
        "test_feature": FEATURES[test_feature],
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "relative_l2_error": float(
            np.linalg.norm(swapped_w - expected)
            / max(np.linalg.norm(expected), 1e-12)
        ),
        "passed_tolerance_1e_5": bool(absolute_error.max() <= 1e-5),
    }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualTabularMLP(nn.Module):
    def __init__(self, p: int, width: int = 128, blocks: int = 3) -> None:
        super().__init__()
        self.input = nn.Linear(p, width)
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, 0.10) for _ in range(blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(torch.nn.functional.gelu(self.input(x)))).squeeze(1)


def train_residual_predictor(
    panel: pd.DataFrame,
    max_epochs: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame, dict]:
    panel = panel.sort_values(["month", "seller_id"]).reset_index(drop=True)
    train_mask = panel["month"].between("2017-01-01", "2017-12-01")
    valid_mask = panel["month"].between("2018-01-01", "2018-04-01")
    test_mask = panel["month"].between("2018-05-01", "2018-07-01")
    preprocessor = RobustPreprocessor(FEATURES).fit(panel.loc[train_mask], copula=False)
    x_train = preprocessor.transform(panel.loc[train_mask]).astype(np.float32)
    x_valid = preprocessor.transform(panel.loc[valid_mask]).astype(np.float32)
    x_test = preprocessor.transform(panel.loc[test_mask]).astype(np.float32)
    y_train = panel.loc[train_mask, "log_gmv_next_month"].to_numpy(np.float32)
    y_valid = panel.loc[valid_mask, "log_gmv_next_month"].to_numpy(np.float32)
    y_test = panel.loc[test_mask, "log_gmv_next_month"].to_numpy(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() or 1.0)
    train_target = (y_train - y_mean) / y_std
    valid_target = (y_valid - y_mean) / y_std

    set_seed(SEED + 6000)
    model = ResidualTabularMLP(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=6, min_lr=2e-5
    )
    loss_function = nn.MSELoss()
    rng = np.random.default_rng(SEED + 6000)
    best_state = None
    best_validation = math.inf
    patience_counter = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(x_train))
        training_losses = []
        for start in range(0, len(order), 512):
            indices = order[start : start + 512]
            xb = torch.from_numpy(x_train[indices]).to(device)
            yb = torch.from_numpy(train_target[indices]).to(device)
            prediction = model(xb)
            loss = loss_function(prediction, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_valid).to(device)
            validation_prediction = model(xv)
            validation_loss = float(
                loss_function(
                    validation_prediction,
                    torch.from_numpy(valid_target).to(device),
                )
                .detach()
                .cpu()
            )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse_scaled": float(np.mean(training_losses)),
                "validation_mse_scaled": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 12:
                break
    if best_state is None:
        raise RuntimeError("Residual predictor failed to train")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        prediction_scaled = (
            model(torch.from_numpy(x_test).to(device)).cpu().numpy()
        )
    prediction = prediction_scaled * y_std + y_mean
    result = metric_row("PyTorch-Residual-MLP", y_test, prediction)
    predictions = pd.DataFrame(
        {
            "seller_id": panel.loc[test_mask, "seller_id"].to_numpy(),
            "month": panel.loc[test_mask, "month"].astype(str).to_numpy(),
            "actual_log_gmv": y_test,
            "predicted_log_gmv": prediction,
        }
    )
    metadata = {
        "architecture": "128-wide residual MLP with 3 residual blocks",
        "optimizer": "AdamW",
        "best_epoch": int(
            pd.DataFrame(history)["validation_mse_scaled"].idxmin() + 1
        ),
        "epochs_run": epoch + 1,
        "best_validation_mse_scaled": best_validation,
        "device": str(device),
    }
    return result, predictions, metadata


def posthoc_evalue_path() -> pd.DataFrame:
    w_frame = pd.read_csv(
        V1_DIR / "results" / "knockoff_runs" / "primary_copula_lasso_w.csv"
    )
    w_matrix = w_frame[FEATURES].to_numpy()
    p = len(FEATURES)
    fixed_alpha_kn = 0.10
    e_runs = np.zeros_like(w_matrix)
    for repetition, w in enumerate(w_matrix):
        threshold = threshold_knockoff_plus(w, fixed_alpha_kn)
        if np.isfinite(threshold):
            selected = w >= threshold
            denominator = 1 + np.sum(w <= -threshold)
            e_runs[repetition, selected] = p / denominator
    mean_evalues = e_runs.mean(axis=0)
    rows = []
    for q in np.arange(0.05, 0.401, 0.025):
        selected = ebh(mean_evalues, float(q))
        rows.append(
            {
                "q_ebh": round(float(q), 3),
                "fixed_alpha_kn": fixed_alpha_kn,
                "selected_count": len(selected),
                "selected_features": "|".join(FEATURES[index] for index in selected),
                "selected_labels_zh": "、".join(
                    FEATURE_LABELS_ZH[FEATURES[index]] for index in selected
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    diagnostics: pd.DataFrame,
    swap: pd.DataFrame,
    deep_selection: pd.DataFrame,
    paired_selection: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "copula_mvr": "#2A6F97",
        "gaussian_mvr": "#8D99AE",
        "deep_knockoff_raw": "#C44536",
        "deep_knockoff_calibrated": "#287271",
    }
    labels = {
        "copula_mvr": "Copula-MVR",
        "gaussian_mvr": "Gaussian-MVR",
        "deep_knockoff_raw": "Deep raw",
        "deep_knockoff_calibrated": "Deep calibrated",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics_to_plot = [
        ("mean_marginal_ks", "Mean marginal KS"),
        ("covariance_relative_error", "Covariance relative error"),
        ("swap_classifier_auc_mean", "Mean swap-classifier AUC"),
        ("coverage_90_mean", "Mean 90% marginal coverage"),
    ]
    for ax, (column, title) in zip(axes.ravel(), metrics_to_plot):
        values = diagnostics.set_index("method")[column]
        methods = values.index.tolist()
        ax.bar(
            [labels[item] for item in methods],
            values.to_numpy(),
            color=[palette[item] for item in methods],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v2_generator_diagnostics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in swap.groupby("method"):
        ax.plot(
            group["swap_ratio"],
            group["mmd_rbf"],
            marker="o",
            label=labels[method],
            color=palette[method],
        )
    ax.set_xlabel("Swap ratio")
    ax.set_ylabel("RBF MMD")
    ax.set_title("Held-out exchangeability diagnostic")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v2_swap_mmd.png", dpi=220)
    plt.close(fig)

    v1_primary = pd.read_csv(V1_DIR / "results" / "knockoff_primary.csv")
    v1_primary = v1_primary.loc[v1_primary["q"].eq(0.20), [
        "feature", "selection_frequency"
    ]].rename(columns={"selection_frequency": "Copula-Lasso"})
    deep_q = deep_selection.loc[deep_selection["q"].eq(0.20), [
        "feature", "selection_frequency"
    ]].rename(columns={"selection_frequency": "DeepGen-Lasso"})
    paired_q = paired_selection.loc[paired_selection["q"].eq(0.20), [
        "feature", "selection_frequency"
    ]].rename(columns={"selection_frequency": "Copula-PairedMLP"})
    combined = v1_primary.merge(deep_q, on="feature").merge(paired_q, on="feature")
    combined["max_frequency"] = combined[
        ["Copula-Lasso", "DeepGen-Lasso", "Copula-PairedMLP"]
    ].max(axis=1)
    combined = combined.nlargest(18, "max_frequency").sort_values(
        "max_frequency"
    )
    y_positions = np.arange(len(combined))
    fig, ax = plt.subplots(figsize=(10, 8))
    for offset, column, color in [
        (-0.22, "Copula-Lasso", "#2A6F97"),
        (0.00, "DeepGen-Lasso", "#287271"),
        (0.22, "Copula-PairedMLP", "#C44536"),
    ]:
        ax.barh(
            y_positions + offset,
            combined[column],
            height=0.20,
            label=column,
            color=color,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_LABELS_ZH[item] for item in combined["feature"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Single-run selection frequency")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v2_deep_selection_comparison.png", dpi=220)
    plt.close(fig)

    plot_metrics = metrics.sort_values("rmse_log")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "#C44536" if name == "PyTorch-Residual-MLP" else "#2A6F97"
        for name in plot_metrics["model"]
    ]
    ax.barh(plot_metrics["model"], plot_metrics["rmse_log"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE on log(1+GMV)")
    ax.set_title("V2 out-of-time model comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_v2_predictive_models.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    parser.add_argument(
        "--resume-generator",
        action="store_true",
        help="Reuse the saved generator checkpoint and training history.",
    )
    args = parser.parse_args()
    if args.mode == "pilot":
        config = RunConfig(
            mode="pilot",
            generator_epochs=6,
            generator_patience=3,
            generator_repetitions=3,
            paired_mlp_repetitions=2,
            paired_mlp_epochs=8,
            predictor_epochs=10,
            diagnostic_permutations=9,
        )
        result_dir = RESULTS_DIR / "pilot"
        figure_dir = FIGURES_DIR / "pilot"
        log_dir = LOG_DIR / "pilot"
        model_dir = MODEL_DIR / "pilot"
    else:
        config = RunConfig(
            mode="full",
            generator_epochs=90,
            generator_patience=12,
            generator_repetitions=30,
            paired_mlp_repetitions=20,
            paired_mlp_epochs=100,
            predictor_epochs=180,
            diagnostic_permutations=99,
        )
        result_dir = RESULTS_DIR
        figure_dir = FIGURES_DIR
        log_dir = LOG_DIR
        model_dir = MODEL_DIR
    for directory in (result_dir, figure_dir, log_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)
    set_seed(SEED)
    device = get_device()
    logging.info("V2 deep analysis mode=%s device=%s", args.mode, device)

    panel = pd.read_csv(
        V1_DIR / "data_processed" / "seller_month_panel.csv",
        parse_dates=["month"],
    )
    if args.mode == "pilot":
        panel_for_training = panel.sample(
            n=min(3200, len(panel)), random_state=SEED
        ).sort_index()
    else:
        panel_for_training = panel
    base_preprocessor = RobustPreprocessor(FEATURES).fit(
        panel_for_training, copula=False
    )
    x_base = base_preprocessor.transform(panel_for_training).astype(np.float32)
    y = panel_for_training["log_gmv_next_month"].to_numpy(np.float32)

    if args.resume_generator:
        generator, history, _, validation_indices = load_generator(
            x_base, device, model_dir, result_dir
        )
    else:
        generator, history, _, validation_indices = train_generator(
            x_base, config, device, model_dir
        )
    history.to_csv(result_dir / "deep_generator_training_history.csv", index=False)

    raw_deep = generate_torch(generator, x_base, device, SEED + 100000)
    calibrated_deep = calibrate_marginals(raw_deep, x_base)

    copula_preprocessor = RobustPreprocessor(FEATURES).fit(
        panel_for_training, copula=True
    )
    x_copula = copula_preprocessor.transform(panel_for_training)
    copula_sampler = make_sampler(x_copula)
    copula_knockoff = sample_knockoff(
        copula_sampler, SEED + 110000
    )
    gaussian_sampler = make_sampler(x_base)
    gaussian_knockoff = sample_knockoff(gaussian_sampler, SEED + 120000)

    diagnostics_rows = []
    swap_frames = []
    validation = validation_indices
    for method, real_values, knockoff in [
        ("copula_mvr", x_copula, copula_knockoff),
        ("gaussian_mvr", x_base, gaussian_knockoff),
        ("deep_knockoff_raw", x_base, raw_deep),
        ("deep_knockoff_calibrated", x_base, calibrated_deep),
    ]:
        summary, swap_frame = full_diagnostics(
            real_values[validation],
            knockoff[validation],
            method,
            config,
        )
        diagnostics_rows.append(summary)
        swap_frames.append(swap_frame)
        logging.info("Diagnostics %s: %s", method, summary)
    diagnostics = pd.DataFrame(diagnostics_rows)
    swap_diagnostics = pd.concat(swap_frames, ignore_index=True)
    diagnostics.to_csv(result_dir / "deep_generator_diagnostics.csv", index=False)
    swap_diagnostics.to_csv(
        result_dir / "deep_swap_diagnostics.csv", index=False
    )

    deep_w = []
    alpha = load_v1_lasso_alpha()
    logging.info(
        "Using fixed V1 primary Lasso alpha %.12f for deep-generator comparison",
        alpha,
    )
    for repetition in range(config.generator_repetitions):
        raw = generate_torch(
            generator,
            x_base,
            device,
            SEED + 130000 + repetition,
        )
        xk = calibrate_marginals(raw, x_base)
        deep_w.append(lasso_w_fixed_alpha(x_base, xk, y, alpha))
        logging.info(
            "Deep-generator Lasso repetition=%s/%s",
            repetition + 1,
            config.generator_repetitions,
        )
    deep_w_matrix = np.asarray(deep_w)
    pd.DataFrame(deep_w_matrix, columns=FEATURES).to_csv(
        result_dir / "deep_generator_lasso_w.csv", index_label="repetition"
    )
    deep_selection = summarize_w(
        deep_w_matrix, "deep_knockoff_calibrated_lasso"
    )
    deep_selection.to_csv(
        result_dir / "deep_generator_selection.csv", index=False
    )

    paired_w = []
    paired_metadata = []
    first_paired_xk = None
    for repetition in range(config.paired_mlp_repetitions):
        xk = sample_knockoff(
            copula_sampler, SEED + 140000 + repetition
        ).astype(np.float32)
        if first_paired_xk is None:
            first_paired_xk = xk.copy()
        w, metadata = train_pairwise_mlp(
            x_copula.astype(np.float32),
            xk,
            y,
            SEED + 150000 + repetition,
            config.paired_mlp_epochs,
            device,
        )
        paired_w.append(w)
        paired_metadata.append(metadata)
        logging.info(
            "Paired MLP repetition=%s validation=%.5f epochs=%s",
            repetition + 1,
            metadata["best_validation_mse"],
            metadata["epochs"],
        )
    paired_w_matrix = np.asarray(paired_w)
    pd.DataFrame(paired_w_matrix, columns=FEATURES).to_csv(
        result_dir / "paired_mlp_w.csv", index_label="repetition"
    )
    paired_selection = summarize_w(
        paired_w_matrix, "copula_knockoff_paired_mlp"
    )
    paired_selection.to_csv(
        result_dir / "paired_mlp_selection.csv", index=False
    )
    (result_dir / "paired_mlp_training.json").write_text(
        json.dumps(paired_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    antisymmetry = paired_mlp_antisymmetry_check(
        x_copula.astype(np.float32),
        first_paired_xk,
        y,
        config.paired_mlp_epochs,
    )
    (result_dir / "paired_mlp_antisymmetry.json").write_text(
        json.dumps(antisymmetry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Paired MLP antisymmetry check: %s", antisymmetry)

    residual_metric, residual_predictions, residual_metadata = (
        train_residual_predictor(panel, config.predictor_epochs, device)
    )
    v1_metrics = pd.read_csv(V1_DIR / "results" / "predictive_model_metrics.csv")
    metrics = pd.concat(
        [v1_metrics, pd.DataFrame([residual_metric])],
        ignore_index=True,
    )
    metrics.to_csv(result_dir / "v2_predictive_model_metrics.csv", index=False)
    residual_predictions.to_csv(
        result_dir / "residual_mlp_test_predictions.csv", index=False
    )
    (result_dir / "residual_mlp_metadata.json").write_text(
        json.dumps(residual_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    path = posthoc_evalue_path()
    path.to_csv(result_dir / "posthoc_evalue_path.csv", index=False)

    make_figures(
        diagnostics,
        swap_diagnostics,
        deep_selection,
        paired_selection,
        metrics,
        figure_dir,
    )
    diagnostic_index = diagnostics.set_index("method")
    copula_row = diagnostic_index.loc["copula_mvr"]
    deep_row = diagnostic_index.loc["deep_knockoff_calibrated"]
    diagnostic_rule = {
        "mean_marginal_ks_max": max(
            0.05, 1.5 * float(copula_row["mean_marginal_ks"])
        ),
        "swap_classifier_auc_max": min(
            0.75, float(copula_row["swap_classifier_auc_mean"]) + 0.10
        ),
        "covariance_relative_error_max": max(
            0.20, 1.5 * float(copula_row["covariance_relative_error"])
        ),
        "mean_real_knockoff_correlation_max": 0.85,
        "swap_mmd_rejection_rate_max": 0.40,
    }
    deep_valid = bool(
        deep_row["mean_marginal_ks"]
        <= diagnostic_rule["mean_marginal_ks_max"]
        and deep_row["swap_classifier_auc_mean"]
        <= diagnostic_rule["swap_classifier_auc_max"]
        and deep_row["covariance_relative_error"]
        <= diagnostic_rule["covariance_relative_error_max"]
        and abs(deep_row["mean_real_knockoff_correlation"])
        <= diagnostic_rule["mean_real_knockoff_correlation_max"]
        and deep_row["swap_mmd_rejection_rate_5pct"]
        <= diagnostic_rule["swap_mmd_rejection_rate_max"]
    )
    metadata = {
        "config": asdict(config),
        "seed": SEED,
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "n": len(panel_for_training),
        "p": len(FEATURES),
        "deep_generator_lasso_alpha": alpha,
        "deep_generator_passed_prespecified_diagnostics": deep_valid,
        "diagnostic_rule": diagnostic_rule,
        "paired_mlp_antisymmetry": antisymmetry,
        "v1_panel_sha256": json.loads(
            (V1_DIR / "results" / "data_audit.json").read_text()
        )["panel_sha256"],
    }
    (result_dir / "v2_reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("V2 deep analysis completed: %s", metadata)


if __name__ == "__main__":
    main()
