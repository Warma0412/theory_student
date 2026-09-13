"""Shared, outcome-blind preprocessing and reproducible experiment utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from build_panel import FEATURES, ROOT

PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
KS = PROTOCOL["selection"]["budgets"]
SEEDS = PROTOCOL["tuning"]["deep_fit_seeds"]
LOG_COLUMNS = [i for i, f in enumerate(FEATURES) if f.startswith("avg_")
               and f not in ("avg_review_score", "avg_estimate_gap_days")]
LOG_COLUMNS += [FEATURES.index(f) for f in
                ("item_count", "order_count", "category_count", "unique_customer_count", "buyer_state_diversity")]


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else
                              x.item() if isinstance(x, np.generic) else str(x)))
    tmp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class Preprocessor:
    def __init__(self, copula=False, business=True):
        self.copula = copula
        self.business = business

    def raw(self, x):
        x = np.asarray(x, dtype=float).copy()
        if self.business:
            x[:, LOG_COLUMNS] = np.log1p(np.maximum(x[:, LOG_COLUMNS], 0))
        x[~np.isfinite(x)] = np.nan
        return x

    def fit(self, x):
        x = self.raw(x)
        self.low = np.nanquantile(x, .01, axis=0)
        self.high = np.nanquantile(x, .99, axis=0)
        self.med = np.nanmedian(x, axis=0)
        self.med = np.nan_to_num(self.med)
        self.low = np.where(np.isfinite(self.low), self.low, self.med)
        self.high = np.where(np.isfinite(self.high), self.high, self.med)
        x = np.where(np.isnan(x), self.med, np.clip(x, self.low, self.high))
        self.sorted = np.sort(x, axis=0)
        self.scaler = StandardScaler().fit(self._rank(x) if self.copula else x)
        return self

    def _rank(self, x):
        n, p = self.sorted.shape
        out = np.empty_like(x)
        # The empirical CDF is fitted on training rows, not recomputed on a test batch.
        for j in range(p):
            lo = np.searchsorted(self.sorted[:, j], x[:, j], side="left")
            hi = np.searchsorted(self.sorted[:, j], x[:, j], side="right")
            jitter = np.random.default_rng(90210 + j).random(len(x))
            u = (lo + (hi - lo) * jitter) / n
            out[:, j] = norm.ppf(np.clip(u, .5 / n, 1 - .5 / n))
        return out

    def transform(self, x):
        x = self.raw(x)
        x = np.where(np.isnan(x), self.med, np.clip(x, self.low, self.high))
        x = self._rank(x) if self.copula else x
        out = self.scaler.transform(x).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite transformed predictors")
        return out


def rank(score, secondary=None):
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite feature scores")
    if secondary is None:
        secondary = np.zeros_like(score)
    return np.lexsort((np.arange(len(score)), -np.asarray(secondary), -score))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def threshold(w, q=.1):
    for t in np.unique(np.abs(w[np.abs(w) > 1e-12])):
        if (1 + np.sum(w <= -t)) / max(1, np.sum(w >= t)) <= q:
            return float(t)
    return math.inf


def ebh(e, q=.2):
    e = np.asarray(e)
    ix = rank(e)
    eligible = np.flatnonzero(e[ix] >= len(e) / (q * np.arange(1, len(e) + 1)))
    return ix[:eligible[-1] + 1] if len(eligible) else np.array([], dtype=int)


def aggregate_w(w):
    e = np.zeros_like(w)
    for i, row in enumerate(w):
        t = threshold(row)
        if np.isfinite(t):
            e[i, row >= t] = len(row) / (1 + np.sum(row <= -t))
    avg = e.mean(axis=0)
    return avg, ebh(avg), (e > 0).mean(axis=0)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    return {
        "rmse_log": float(mean_squared_error(y, pred) ** .5),
        "mae_log": float(mean_absolute_error(y, pred)),
        "r2_log": float(r2_score(y, pred)),
        "wape_raw": float(np.abs(np.expm1(y) - np.maximum(0, np.expm1(pred))).sum()
                          / max(np.expm1(y).sum(), 1e-12)),
    }


def load_data(seller_ids=None):
    df = pd.read_parquet(ROOT / "data_processed/seller_month_asof.parquet")
    if seller_ids is not None:
        df = df.loc[df.seller_id.isin(seller_ids)]
    return df


def split_arrays(df, fit_splits=("train",), valid_split="tune"):
    tr = df.loc[df.split.isin(fit_splits)]
    va = df.loc[df.split.eq(valid_split)]
    return (tr[FEATURES].to_numpy(float), tr.log_gmv_next_month.to_numpy(float),
            va[FEATURES].to_numpy(float), va.log_gmv_next_month.to_numpy(float),
            tr.seller_id.to_numpy(), va.seller_id.to_numpy())


def make_sampling_manifest():
    df = load_data()
    sellers = np.sort(df.loc[df.split.ne("test"), "seller_id"].unique())
    out = []
    for group in range(30):
        shuffled = np.random.default_rng(20260908 + group).permutation(sellers)
        a, b = np.array_split(shuffled, 2)
        for half, ids in (("A", a), ("B", b)):
            out.append(dict(run=f"half_{group:02d}_{half}", group=group, fraction=.5,
                            half=half, sellers=sorted(ids.tolist())))
    for fraction in (.7, .8):
        for group in range(20):
            ids = np.random.default_rng(20261908 + group).permutation(sellers)[:int(fraction * len(sellers))]
            out.append(dict(run=f"f{int(fraction * 100)}_{group:02d}", group=group,
                            fraction=fraction, half="", sellers=sorted(ids.tolist())))
    for row in out:
        row["seller_hash"] = hashlib.sha256("\n".join(row["sellers"]).encode()).hexdigest()
    save_json(ROOT / "data_processed/sampling_manifest.json", out)
    return out
