"""Remove incidental input-order effects from downstream subset evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json

from analyze_study import ALL, SCENES, source, runfile
from run_study import STUDY, V7, RESULT, read, SEEDS
from common import FEATURES, Preprocessor, load_data, metrics, save_json, digest_array
from models import fit_neural, neural_predict, xgb_model
import numpy as np
import pandas as pd


def real_evaluation(evaluator, shard=0, shards=1):
    frame = load_data()
    tr = frame.loc[frame.split.isin(["train", "tune"])]
    va, te = frame.loc[frame.split.eq("rank")], frame.loc[frame.split.eq("test")]
    pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
    allx, allv, allt = [pre.transform(df[FEATURES].to_numpy(float)) for df in (tr, va, te)]
    y, yv = tr.log_gmv_next_month.to_numpy(), va.log_gmv_next_month.to_numpy()
    config_name = "xgboost_shap" if evaluator == "xgboost" else "tabm"
    config = read(V7 / f"results/configs/{config_name}.json")["params"]
    for method in ALL + ["all_features"]:
        for k in ([31] if method == "all_features" else [8, 12]):
            cols = list(range(31)) if k == 31 else sorted(
                read(runfile(method, "reference" if k == 12 else "reference_k8"))["order"][:k])
            owner = int(hashlib.sha256(json.dumps(cols).encode()).hexdigest()[:8], 16) % shards
            if owner != shard:
                continue
            destination = RESULT / "canonical_predictions" / f"{method}_{evaluator}_k{k}.csv"
            key = "_".join(map(str, cols)) + "_" + evaluator
            cachefile = RESULT / "canonical_cache" / f"{key}.npz"
            if destination.exists() and destination.with_suffix(".json").exists():
                continue
            if cachefile.exists():
                arrays = np.load(cachefile)
                pred, val = arrays["prediction"], arrays["validation"]
            else:
                x, xv, xt = allx[:, cols], allv[:, cols], allt[:, cols]
                predictions, validations = [], []
                for seed in SEEDS:
                    if evaluator == "xgboost":
                        model = xgb_model(config, seed).fit(x, y)
                        validations.append(model.predict(xv))
                        model = xgb_model(config, seed).fit(np.concatenate([x, xv]), np.r_[y, yv])
                        predictions.append(model.predict(xt))
                    else:
                        model, meta = fit_neural(x, y, xv, yv, config, "tabm", seed)
                        validations.append(neural_predict(model, xv))
                        best_epoch = min(meta["history"], key=lambda r: r[2])[0]
                        model, _ = fit_neural(np.concatenate([x, xv]), np.r_[y, yv], xv, yv,
                                             config | {"epochs": best_epoch, "fixed_epochs": True}, "tabm", seed)
                        predictions.append(neural_predict(model, xt))
                pred, val = np.mean(predictions, 0), np.mean(validations, 0)
                cachefile.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cachefile, prediction=pred, validation=val)
            out = te[["row_id", "seller_id", "target_month", "log_gmv_next_month"]].copy()
            out["prediction"] = pred
            destination.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(destination, index=False)
            save_json(destination.with_suffix(".json"), {"method": method, "evaluator": evaluator, "k": k,
                "features": [FEATURES[j] for j in cols], "test": metrics(te.log_gmv_next_month.to_numpy(), pred),
                "validation": metrics(yv, val), "canonical_columns": True, "fit_seeds": SEEDS,
                "test_labels_not_used_for_tuning": True, "test_previously_observed": True,
                "identical_subset_identical_predictions": True})
            print("canonical", method, evaluator, k, flush=True)


def simulation_evaluation(shard, shards):
    for index, (scene, rep) in enumerate((s, r) for s in SCENES for r in range(100)):
        if index % shards != shard:
            continue
        destination = RESULT / "canonical_simulation_evaluation" / f"{scene}_{rep:03d}.json"
        if destination.exists():
            continue
        paths = {m: source(m) / "simulation_runs" / m / f"{scene}_{rep:03d}.json" for m in ALL}
        if not all(p.exists() for p in paths.values()):
            continue
        raw = dict(np.load(V7 / "data_processed/simulations" / f"{scene}_{rep:03d}.npz"))
        pre = Preprocessor(business=False).fit(raw["x"][:1200])
        x, xt = pre.transform(raw["x"][:1200]), pre.transform(raw["x"][1600:])
        cache, rows = {}, {}
        for method, path in paths.items():
            selection = read(path)
            assert selection["status"] == "ok"
            cols = tuple(sorted(selection["order"][:12]))
            if cols not in cache:
                model = xgb_model({"trees": 140, "depth": 3, "lr": .05}, 11).fit(x[:, cols], raw["y"][:1200])
                cache[cols] = float(np.mean((model.predict(xt[:, cols]) - raw["y"][1600:])**2)**.5)
            rows[method] = {"rmse": cache[cols], "columns": list(cols)}
        save_json(destination, {"scene": scene, "rep": rep, "methods": rows, "unique_subsets": len(cache),
                               "data_hash": digest_array(raw["x"]), "common_preprocessing": True})
        print("canonical simulation", scene, rep, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["real", "simulate"])
    parser.add_argument("--evaluator", choices=["xgboost", "tabm"], default="xgboost")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "real":
        real_evaluation(args.evaluator, args.shard, args.shards)
    else:
        simulation_evaluation(args.shard, args.shards)
