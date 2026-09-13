"""Swap-canonical held-out loss contrasts for fixed independent predictors."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT.parent / "v7"
sys.path.insert(0, str(V7 / "code"))

from common import seed_all, rank, save_json, digest_file, digest_array, Preprocessor
from models import xgb_model, fit_neural, neural_predict, deep_lasso_scores
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import ElasticNet
from xgboost import DMatrix

P = json.loads((ROOT / "protocol.json").read_text())
SEEDS = P["predictors"]["seeds"]


def threshold(w, q=.2):
    candidates = np.unique(np.abs(w[np.abs(w) > 1e-12]))
    for cutoff in candidates:
        if (1 + np.sum(w <= -cutoff)) / max(1, np.sum(w >= cutoff)) <= q:
            return float(cutoff)
    return np.inf


def selection(score, native=True, meta=None):
    score = np.asarray(score, dtype=float)
    assert np.isfinite(score).all()
    out = {"score": score.tolist(), "order": rank(score).tolist(),
           "metadata": meta or {}, "native": None, "native_q10": None}
    if native:
        out["native"] = np.flatnonzero(score >= threshold(score, .2)).tolist()
        out["native_q10"] = np.flatnonzero(score >= threshold(score, .1)).tolist()
    return out


def gaussian_knockoff(x, covariance, seed, mean=None):
    p = x.shape[1]
    mu = np.zeros(p) if mean is None else mean
    inv = np.linalg.inv(covariance)
    s = min(1.0, 1.8 * np.linalg.eigvalsh(covariance).min())
    if s <= 0:
        raise ValueError("Covariance is not positive definite")
    noise_cov = 2 * s * np.eye(p) - s * s * inv
    noise_cov = (noise_cov + noise_cov.T) / 2
    chol = np.linalg.cholesky(noise_cov)
    noise = np.random.default_rng(seed).normal(size=x.shape)
    xk = mu + (x - mu) @ (np.eye(p) - s * inv) + noise @ chol.T
    return xk, {"s": s, "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min())}


class TrainingCopula:
    """Approximate generator only; no Olist exchangeability guarantee."""
    def fit(self, x):
        self.sorted = np.sort(x, axis=0).astype(float)
        z = self.gaussianize(x)
        self.mean = z.mean(0)
        self.covariance = LedoitWolf().fit(z).covariance_
        return self

    def gaussianize(self, x):
        z = np.zeros_like(x, dtype=float)
        n = len(self.sorted)
        for j in range(x.shape[1]):
            lo = np.searchsorted(self.sorted[:, j], x[:, j], side="left")
            hi = np.searchsorted(self.sorted[:, j], x[:, j], side="right")
            z[:, j] = ndtri(np.clip((lo + hi) / (2 * n), .5 / n, 1 - .5 / n))
        return z

    def generate(self, x, seed):
        z = self.gaussianize(x)
        zk, meta = gaussian_knockoff(z, self.covariance, seed, self.mean)
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            q = np.clip(np.floor(ndtr(zk[:, j]) * len(self.sorted)).astype(int), 0, len(self.sorted) - 1)
            out[:, j] = self.sorted[q, j]
        return out, meta | {"validity": "estimated copula approximate only"}

    def generate_mvr(self, x, seed):
        from knockpy.knockoffs import GaussianSampler
        seed_all(seed)
        sampler = GaussianSampler(X=self.gaussianize(x), mu=self.mean, Sigma=self.covariance, method="mvr")
        zk = sampler.sample_knockoffs()
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            q = np.clip(np.floor(ndtr(zk[:, j]) * len(self.sorted)).astype(int), 0, len(self.sorted) - 1)
            out[:, j] = self.sorted[q, j]
        return out, {"s_diagonal": np.diag(sampler.S).tolist(), "validity": "estimated copula approximate only",
                     "method": "existing MVR", "posthoc_extension": True}


def grouped_statistics(delta, groups):
    _, index = np.unique(groups, return_inverse=True)
    n = np.bincount(index)
    means = np.zeros((len(n), delta.shape[1]))
    for j in range(delta.shape[1]):
        means[:, j] = np.bincount(index, weights=delta[:, j]) / n
    score = means.mean(0)
    se = means.std(0, ddof=1) / np.sqrt(len(n))
    return score, score / (se + 1e-8), se


def context_delta(predict, x, xk, y, contexts):
    n, p = x.shape
    contrast = np.zeros((n, p), dtype=np.float64)
    for context in contexts:
        for j in range(p):
            original = context.copy()
            artificial = context.copy()
            original[:, j] = x[:, j]
            artificial[:, j] = xk[:, j]
            pair = predict(np.concatenate([original, artificial]))
            contrast[:, j] += ((pair[n:] - y)**2 - (pair[:n] - y)**2) / len(contexts)
    return contrast


def contexts_for(x, xk, draws=8, seed=772):
    rng = np.random.default_rng(seed)
    lo, hi = np.minimum(x, xk), np.maximum(x, xk)
    contexts = []
    for _ in range(draws // 2):
        mask = rng.random(x.shape) < .5
        contexts.extend([np.where(mask, lo, hi), np.where(mask, hi, lo)])
    return contexts


def loss_family(predict, x, xk, y, groups, seed=772, draws=8):
    raw = context_delta(predict, x, xk, y, [x])
    middle = context_delta(predict, x, xk, y, [(x + xk) / 2])
    orbit = context_delta(predict, x, xk, y, contexts_for(x, xk, draws, seed))
    r, _, _ = grouped_statistics(raw, groups)
    m, _, _ = grouped_statistics(middle, groups)
    o, t, se = grouped_statistics(orbit, groups)
    return {
        "naive": selection(r, meta={"full_sign_flip_property": False, "nominal_threshold_diagnostic_only": True}),
        "midpoint": selection(m, meta={"full_sign_flip_property": True}),
        "orbit": selection(o, meta={"full_sign_flip_property": True, "contexts": draws}),
        "orbit_student": selection(t, meta={"full_sign_flip_property": True, "contexts": draws,
                                            "mean_contrast": o.tolist(), "se": se.tolist(),
                                            "not_a_t_test": True}),
    }


def fit_predictors(x, y, nfit, seeds, save_dir=None):
    config_xgb = json.loads((V7 / "results/configs/xgboost_shap.json").read_text())["params"]
    trees = [xgb_model(config_xgb, seed).fit(x, y) for seed in seeds]
    models, metadata = {}, {}
    for kind in ("tabm", "deep_lasso"):
        config = json.loads((V7 / f"results/configs/{kind}.json").read_text())["params"]
        fits, histories = [], []
        for seed in seeds:
            first, meta = fit_neural(x[:nfit], y[:nfit], x[nfit:], y[nfit:], config, kind, seed)
            epoch = min(meta["history"], key=lambda r: r[2])[0]
            refit, finalmeta = fit_neural(x, y, x[nfit:], y[nfit:],
                                         config | {"epochs": epoch, "fixed_epochs": True,
                                                   "patience": epoch + 1}, kind, seed)
            fits.append(refit)
            histories.append({"seed": seed, "selected_epoch": epoch, "initial": meta, "refit": finalmeta})
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": refit.state_dict(), "config": config, "y_mean": refit.y_mean,
                            "y_std": refit.y_std, "history": histories[-1]}, save_dir / f"{kind}_{seed}.pt")
        models[kind] = fits
        metadata[kind] = histories
    if save_dir:
        for i, model in enumerate(trees):
            model.save_model(save_dir / f"xgb_{seeds[i]}.json")
    return trees, models, metadata


def permutation_score(predict, x, y, seed=11):
    base = np.mean((predict(x) - y)**2)
    score = np.zeros(x.shape[1])
    rng = np.random.default_rng(seed)
    for j in range(x.shape[1]):
        for _ in range(3):
            copy = x.copy()
            copy[:, j] = rng.permutation(copy[:, j])
            score[j] += (np.mean((predict(copy) - y)**2) - base) / 3
    return score


def execute_methods(x, y, xk, ntrain, nfit, groups, seeds=SEEDS, save_dir=None):
    train, target = x[:ntrain], y[:ntrain]
    infer, response = x[ntrain:], y[ntrain:]
    mean, std = target.mean(), max(target.std(), 1e-6)
    # Conditional on the training sample, only inference pairs remain exchangeable.
    joint = np.column_stack([infer, xk[ntrain:]])
    sym_mean = (joint[:, :x.shape[1]].mean(0) + joint[:, x.shape[1]:].mean(0)) / 2
    pooled = np.concatenate([infer, xk[ntrain:]])
    sym_std = np.maximum(pooled.std(0), 1e-6)
    scaled = (joint - np.r_[sym_mean, sym_mean]) / np.r_[sym_std, sym_std]
    linear = ElasticNet(alpha=.01, l1_ratio=.999, tol=1e-7, max_iter=20000).fit(scaled, (response - mean) / std)
    coef = np.abs(linear.coef_)
    out = {"lasso_pair_q20": selection(coef[:x.shape[1]] - coef[x.shape[1]:])}
    linear_predictor = ElasticNet(alpha=.01, l1_ratio=.999, max_iter=20000).fit(train, (target - mean) / std)
    predict_linear = lambda z: linear_predictor.predict(z) * std + mean
    linear_delta = context_delta(predict_linear, infer, xk[ntrain:], response,
                                contexts_for(infer, xk[ntrain:], P["orbit_contexts"], seeds[0] + 771))
    _, linear_student, _ = grouped_statistics(linear_delta, groups[ntrain:])
    out["linear_orbit_student"] = selection(linear_student)
    trees, networks, meta = fit_predictors(train, target, nfit, seeds, save_dir)
    predict_tree = lambda z: np.mean([model.predict(z) for model in trees], 0)
    predict_tabm = lambda z: np.mean([neural_predict(model, z) for model in networks["tabm"]], 0)
    shap = np.mean([np.abs(model.get_booster().predict(DMatrix(infer),
                        pred_contribs=True)[:, :-1]).mean(0) for model in trees], 0)
    out["xgb_shap"] = selection(shap, native=False)
    out["xgb_pfi"] = selection(permutation_score(predict_tree, infer, response, seeds[0]), native=False)
    out["tabm_pfi"] = selection(permutation_score(predict_tabm, infer, response, seeds[0]), native=False)
    dlscore = np.mean([deep_lasso_scores(m, infer, response) for m in networks["deep_lasso"]], 0)
    out["deep_lasso"] = selection(dlscore, native=False)
    for name, predict in (("xgb", predict_tree), ("tabm", predict_tabm)):
        for mode, record in loss_family(predict, infer, xk[ntrain:], response,
                                       groups[ntrain:], seed=seeds[0] + 771,
                                       draws=P["orbit_contexts"]).items():
            out[f"{name}_{mode}"] = record
    return out, meta
