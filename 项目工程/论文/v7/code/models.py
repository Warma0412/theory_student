"""Feature selectors with paper implementations and explicit research adaptations."""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import time
import types
from dataclasses import dataclass
from pathlib import Path

from common import ROOT, SEEDS, aggregate_w, rank, seed_all
import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from xgboost import XGBRegressor, DMatrix
from knockpy.knockoffs import GaussianSampler


def module_from_file(name, path, excluded_imports=()):
    tree = ast.parse(Path(path).read_text())
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.ImportFrom)
                and any((node.module or "").startswith(s) for s in excluded_imports))
    ]
    module = types.ModuleType(name)
    module.__dict__.update({"FeatureEvaluator": object, "info": lambda *x: None})
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


DL = module_from_file("official_deep_lasso", ROOT / "vendor/deep_lasso/deep_lasso.py")


@dataclass
class Selection:
    score: np.ndarray
    order: np.ndarray
    native: list[int] | None
    metadata: dict
    model: object = None


def selection(score, native=None, metadata=None, model=None, secondary=None):
    return Selection(np.asarray(score), rank(score, secondary),
                     None if native is None else list(map(int, native)),
                     metadata or {}, model)


def xgb_model(params, seed=11):
    return XGBRegressor(
        n_estimators=int(params.get("trees", 220)), max_depth=int(params.get("depth", 3)),
        learning_rate=float(params.get("lr", .04)), min_child_weight=10,
        subsample=.8, colsample_bytree=.8, reg_lambda=2.,
        objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=seed,
    )


def fit_neural(x, y, xv, yv, params, kind="deep_lasso", seed=11):
    seed_all(seed)
    start = time.monotonic()
    p = x.shape[1]
    width, layers = int(params.get("width", 64)), int(params.get("layers", 2))
    if kind == "tabm":
        from tabm import TabM
        model = TabM.make(n_num_features=p, d_out=1, k=8,
                          n_blocks=layers, d_block=width, dropout=.1)
    else:
        parts = []
        for _ in range(layers):
            parts += [nn.Linear(p, width), nn.ReLU(), nn.Dropout(.1)]
            p = width
        model = nn.Sequential(*parts, nn.Linear(width, 1))
    # CPU is deliberate: input-gradient penalties require double backward.
    device = torch.device("cpu")
    model.to(device)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float32)
    xvt, yvt = torch.as_tensor(xv), torch.as_tensor(yv, dtype=torch.float32)
    mean, std = float(yt.mean()), max(float(yt.std(unbiased=False)), 1e-6)
    yt, yvt = (yt - mean) / std, (yvt - mean) / std
    opt = torch.optim.AdamW(model.parameters(), lr=params.get("lr", .001), weight_decay=2e-4)
    best, patience, state, history = np.inf, 0, None, []
    regularization = params.get("regularization", .1) if kind == "deep_lasso" else 0
    max_epochs = int(params.get("epochs", 100))
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for idx in torch.randperm(len(xt)).split(512):
            xb = xt[idx].clone().requires_grad_(regularization > 0)
            pred = model(xb)
            loss = (pred.squeeze(-1) - (yt[idx, None] if pred.ndim == 3 else yt[idx])).square().mean()
            if regularization > 0:
                loss = loss + regularization * DL.deep_lasso_regularizer(loss, xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(b) for b in xvt.split(1024)])
            pv = out.mean(1).squeeze(-1) if out.ndim == 3 else out.squeeze(-1)
            valid = float((pv - yvt).square().mean())
        if not np.isfinite(valid):
            raise ValueError(f"{kind}: nonfinite validation loss")
        history.append([epoch + 1, float(np.mean(losses)), valid])
        if params.get("fixed_epochs", False) or valid < best - 1e-6:
            best, patience = valid, 0
            state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= int(params.get("patience", 12)):
            break
    model.load_state_dict(state)
    model.eval()
    model.y_mean, model.y_std = mean, std
    meta = {"kind": kind, "seed": seed, "epochs": len(history), "best_validation_mse_scaled": best,
            "train_seconds": time.monotonic() - start, "parameters": sum(p.numel() for p in model.parameters()),
            "history": history, "device": str(device)}
    return model, meta


def neural_predict(model, x):
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(b) for b in torch.as_tensor(x, dtype=torch.float32).split(1024)])
        preds = preds.mean(1).squeeze(-1) if preds.ndim == 3 else preds.squeeze(-1)
    return preds.numpy() * model.y_std + model.y_mean


def deep_lasso_scores(model, x, y):
    grads = []
    target = (np.asarray(y) - model.y_mean) / model.y_std
    for begin in range(0, len(x), 512):
        xb = torch.as_tensor(x[begin:begin + 512]).clone().requires_grad_(True)
        yb = torch.as_tensor(target[begin:begin + 512], dtype=torch.float32)
        loss = (model(xb).squeeze(-1) - yb).square().mean()
        grad, = torch.autograd.grad(loss, xb)
        grads.append(grad.detach())
    return torch.cat(grads).square().sum(0).sqrt().numpy()


def permutation_scores(predict, x, y, seed, repeats=3):
    rng = np.random.default_rng(seed)
    base = np.mean((predict(x) - y) ** 2)
    scores = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        for _ in range(repeats):
            z = x.copy()
            z[:, j] = z[rng.permutation(len(z)), j]
            scores[j] += (np.mean((predict(z) - y) ** 2) - base) / repeats
    return scores


def lasso_competition(x, xk, y, alpha=.01):
    # A tiny symmetric ridge term makes the convex minimizer unique under ties.
    # l1_ratio=0.999 is disclosed as an Elastic-Net approximation to Lasso.
    model = ElasticNet(alpha=alpha, l1_ratio=.999, max_iter=10000, tol=1e-6).fit(
        np.column_stack([x, xk]), y
    )
    p = x.shape[1]
    return np.abs(model.coef_[:p]) - np.abs(model.coef_[p:])


def knockoff_select(x, y, seed=11, draws=60, generator=None):
    start = time.monotonic()
    ys = (y - y.mean()) / max(y.std(), 1e-6)
    if generator is None:
        sigma = LedoitWolf().fit(x).covariance_
        sampler = GaussianSampler(X=np.asarray(x, dtype=float), mu=x.mean(0), Sigma=sigma, method="mvr")
    w = []
    for i in range(draws):
        seed_all(seed + i)
        xk = sampler.sample_knockoffs() if generator is None else generator(x, seed + i)
        w.append(lasso_competition(x, xk, ys))
    w = np.asarray(w)
    score, native, freq = aggregate_w(w)
    return selection(score, native, {"w": w.tolist(), "frequency": freq.tolist(),
                                     "seconds": time.monotonic() - start,
                                     "draws": draws, "statistic_l1_ratio": .999},
                     secondary=w.mean(0))


def run_selector(method, x, y, xv, yv, sellers, params, seeds=SEEDS, prefix=None):
    start = time.monotonic()
    if method == "copula_mvr":
        result = knockoff_select(x, y, seed=seeds[0], draws=60)
    elif method == "elastic_net":
        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                           max_iter=10000, tol=1e-6).fit(x, y)
        score = np.abs(model.coef_)
        result = selection(score, np.flatnonzero(score > 1e-8), model=model)
    elif method == "stability_selection":
        coefficients = []
        ids = np.unique(sellers)
        for rep in range(20):
            order = np.random.default_rng(seeds[0] + rep).permutation(ids)
            for half in np.array_split(order, 2):
                idx = np.isin(sellers, half)
                m = Lasso(alpha=params["alpha"], max_iter=10000, tol=1e-6).fit(x[idx], y[idx])
                coefficients.append(np.abs(m.coef_))
        coef = np.array(coefficients)
        freq = (coef > 1e-8).mean(0)
        result = selection(freq, np.flatnonzero(freq >= .9),
                           {"inner_complementary_pairs": 20}, secondary=coef.mean(0))
    elif method == "shadow_trees":
        rows, wins = [], []
        for rep in range(8):
            rng = np.random.default_rng(seeds[0] + rep)
            shadow = np.stack([rng.permutation(x[:, j]) for j in range(x.shape[1])], axis=1)
            model = ExtraTreesRegressor(n_estimators=params.get("trees", 160),
                        min_samples_leaf=params.get("leaf", 8), max_features=.7, n_jobs=1,
                        random_state=seeds[0] + rep).fit(np.column_stack([x, shadow]), y)
            real, fake = np.split(model.feature_importances_, 2)
            rows.append(real)
            wins.append(real > fake.max())
        scores, frequency = np.mean(rows, 0), np.mean(wins, 0)
        result = selection(frequency, np.flatnonzero(frequency >= .75),
                           {"inner_draws": 8}, secondary=scores)
    elif method == "xgboost_shap":
        scores = []
        for seed in seeds:
            model = xgb_model(params, seed).fit(x, y)
            scores.append(np.abs(model.get_booster().predict(DMatrix(xv), pred_contribs=True)[:, :-1]).mean(0))
        result = selection(np.mean(scores, 0), model=model)
    elif method in ("deep_lasso", "tabm"):
        scores, trained, metas = [], [], []
        for seed in seeds:
            model, meta = fit_neural(x, y, xv, yv, params, method, seed)
            score = deep_lasso_scores(model, xv, yv) if method == "deep_lasso" else permutation_scores(
                lambda z: neural_predict(model, z), xv, yv, seed)
            scores.append(score)
            trained.append(model)
            metas.append(meta)
            if prefix is not None:
                Path(prefix).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "params": params, "p": x.shape[1],
                            "y_mean": model.y_mean, "y_std": model.y_std, "metadata": meta},
                           str(prefix) + f"_seed{seed}.pt")
        result = selection(np.mean(scores, axis=0), metadata={"fits": metas}, model=trained)
    elif method == "vtfs":
        from vtfs_adapter import vtfs_select
        result = vtfs_select(x, y, xv, yv, params, seeds, prefix)
    elif method == "deepdrk":
        from deepdrk_adapter import deepdrk_select
        result = deepdrk_select(x, y, xv, yv, params, seeds, prefix)
    elif method == "tabpfn_v2":
        from tabpfn import TabPFNRegressor
        if len(x) > 10000:
            raise ValueError("TabPFN v2: full training sample exceeds published 10000-row setting; no downsampling allowed")
        scores = []
        for seed in seeds:
            model = TabPFNRegressor(device="mps", n_estimators=4, random_state=seed,
                                    memory_saving_mode=True, fit_mode="fit_with_cache")
            model.fit(x, y)
            scores.append(permutation_scores(model.predict, xv, yv, seed))
        result = selection(np.mean(scores, 0), model=model)
    else:
        raise ValueError(method)
    result.metadata["total_seconds"] = time.monotonic() - start
    result.metadata["training_rows"] = len(x)
    result.metadata["ranking_rows"] = len(xv)
    result.metadata["seeds"] = list(seeds)
    return result
