"""Post-hoc reference-only generator check; primary equicorrelated runs untouched."""

import json
from pathlib import Path

from paired_loss import ROOT, V7, P, SEEDS, TrainingCopula, loss_family
from common import FEATURES, load_data, Preprocessor, seed_all, save_json, metrics
from models import neural_predict, xgb_model
from knockpy.knockoffs import GaussianSampler
from xgboost import XGBRegressor
from tabm import TabM
import numpy as np
import torch
from scipy.special import ndtr

frame = load_data()
train = frame.loc[frame.split.isin(["train", "tune"])]
infer = frame.loc[frame.split.eq("rank")]
test = frame.loc[frame.split.eq("test")]
pre = Preprocessor().fit(train[FEATURES].to_numpy(float))
xt = pre.transform(train[FEATURES].to_numpy(float))
x = pre.transform(infer[FEATURES].to_numpy(float))
y = infer.log_gmv_next_month.to_numpy()
copula = TrainingCopula().fit(xt)
z = copula.gaussianize(x)
seed_all(P["real_data"]["generator_seed"])
sampler = GaussianSampler(X=z, mu=copula.mean, Sigma=copula.covariance, method="mvr")
zk = sampler.sample_knockoffs()
xk = np.zeros_like(x)
for j in range(x.shape[1]):
    ids = np.clip(np.floor(ndtr(zk[:, j]) * len(copula.sorted)).astype(int), 0, len(copula.sorted) - 1)
    xk[:, j] = copula.sorted[ids, j]
trees, nets = [], []
for seed in SEEDS:
    tree = XGBRegressor()
    tree.load_model(ROOT / "models/reference" / f"xgb_{seed}.json")
    trees.append(tree)
    checkpoint = torch.load(ROOT / "models/reference" / f"tabm_{seed}.pt", weights_only=False)
    params = checkpoint["config"]
    model = TabM.make(n_num_features=31, d_out=1, k=8, n_blocks=params["layers"], d_block=params["width"], dropout=.1)
    model.load_state_dict(checkpoint["state_dict"])
    model.y_mean, model.y_std = checkpoint["y_mean"], checkpoint["y_std"]
    model.eval()
    nets.append(model)
out = {}
for name, pred in (("xgb", lambda v: np.mean([m.predict(v) for m in trees], 0)),
                   ("tabm", lambda v: np.mean([neural_predict(m, v) for m in nets], 0))):
    for mode, record in loss_family(pred, x, xk, y, infer.seller_id.to_numpy(), seed=SEEDS[0] + 771).items():
        out[f"{name}_{mode}"] = record
config = json.loads((V7 / "results/configs/xgboost_shap.json").read_text())["params"]
xdev = np.concatenate([xt, x])
ydev = np.r_[train.log_gmv_next_month.to_numpy(), y]
xeval = pre.transform(test[FEATURES].to_numpy(float))
predictions = []
for method, record in out.items():
    cols = sorted(record["order"][:12])
    prediction = np.mean([xgb_model(config, seed).fit(xdev[:, cols], ydev).predict(xeval[:, cols]) for seed in SEEDS], 0)
    predictions.append({"method": method, "top12_xgb_rmse": metrics(test.log_gmv_next_month, prediction)["rmse_log"],
                        "native_count": len(record["native"]), "top12": cols})
save_json(ROOT / "results/mvr_reference_sensitivity.json", {
    "posthoc_diagnostic": True, "reference_only": True, "all_primary_runs_unchanged": True,
    "prediction_networks_reused_without_retraining": True,
    "generator": "Spector-Janson MVR using knockpy; not a new generator",
    "mean_squared_pair_distance": float(np.mean((x - xk)**2)),
    "pair_equal_fraction": float(np.mean(x == xk)),
    "s_diagonal": np.diag(sampler.S).tolist(),
    "methods": out, "predictions": predictions,
    "no_sample_stability_or_simulation_claim_for_this_sensitivity": True,
    "no_real_fdr_certificate": True})
print("MVR reference sensitivity completed")
for row in predictions:
    print(row["method"], row["top12_xgb_rmse"], row["native_count"])
