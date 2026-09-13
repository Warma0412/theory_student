"""Checkpointed V7 experiments. Commands never overwrite another method's runs."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import platform
import subprocess
import time
import traceback
from pathlib import Path

from common import (ROOT, FEATURES, KS, SEEDS, PROTOCOL, Preprocessor, digest_file,
                    digest_array, load_data, split_arrays, make_sampling_manifest,
                    metrics, rank, save_json, seed_all, jaccard)
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import ExtraTreesRegressor

from models import run_selector, fit_neural, neural_predict, xgb_model

METHODS = ["copula_mvr", "elastic_net", "stability_selection", "shadow_trees",
           "xgboost_shap", "deep_lasso", "tabm", "vtfs", "deepdrk", "tabpfn_v2"]


def config_path(method):
    return ROOT / "results/configs" / f"{method}.json"


def configurations(method):
    if method == "elastic_net":
        return [{"alpha": a, "l1_ratio": l} for a, l in itertools.product(
            (.001, .003, .01, .03, .1, .3), (.5, .9))]
    if method == "stability_selection":
        return [{"alpha": float(a)} for a in np.logspace(-3, -.1, 12)]
    if method == "xgboost_shap":
        return [{"depth": d, "lr": lr, "trees": trees} for d, lr, trees in
                itertools.product((2, 4, 6), (.03, .07), (140, 280))]
    if method == "shadow_trees":
        return [{"trees": trees, "leaf": leaf} for trees, leaf in
                itertools.product((80, 160, 240), (4, 8, 16, 24))]
    if method == "deep_lasso":
        return [{"width": w, "layers": 2, "lr": lr, "regularization": reg}
                for w, lr, reg in itertools.product((64, 128), (.001, .003), (.01, .1, 1.))]
    if method == "tabm":
        return [{"width": w, "layers": layers, "lr": lr}
                for w, layers, lr in itertools.product((64, 128), (2, 3), (.0005, .001, .003))]
    if method == "vtfs":
        return [{"width": 64, "layers": 2, "lr": .001, "epochs": 100}]
    if method == "deepdrk":
        return [{"width": 64, "layers": 2, "lr": .00001, "epochs": 200}]
    return [{}]


def transformed(df, method, fit_splits=("train",), valid_split="tune"):
    x, y, xv, yv, s, sv = split_arrays(df, fit_splits, valid_split)
    pre = Preprocessor(copula=method == "copula_mvr").fit(x)
    mean, std = y.mean(), max(y.std(), 1e-6)
    return pre.transform(x), (y - mean) / std, pre.transform(xv), (yv - mean) / std, s, sv, pre


def tune(method):
    path = config_path(method)
    if path.exists():
        return
    df = load_data()
    x, y, xv, yv, s, sv, pre = transformed(df, method)
    trials = []
    for i, params in enumerate(configurations(method)):
        t = time.monotonic()
        if method == "elastic_net":
            model = ElasticNet(**params, max_iter=10000).fit(x, y)
            pred = model.predict(xv)
        elif method == "stability_selection":
            model = Lasso(**params, max_iter=10000).fit(x, y)
            pred = model.predict(xv)
        elif method == "shadow_trees":
            model = ExtraTreesRegressor(n_estimators=params["trees"], min_samples_leaf=params["leaf"],
                                        max_features=.7, n_jobs=1, random_state=11).fit(x, y)
            pred = model.predict(xv)
        elif method == "xgboost_shap":
            model = xgb_model(params).fit(x, y)
            pred = model.predict(xv)
        elif method in ("deep_lasso", "tabm"):
            model, meta = fit_neural(x, y, xv, yv, params, method, 11)
            pred = neural_predict(model, xv)
        else:
            # Native default settings are not retuned using final discoveries.
            trials.append({"trial": i, "params": params, "rmse_scaled": None,
                           "selection": "paper/default computational adaptation; no outcome-based search"})
            break
        trials.append({"trial": i, "params": params, "rmse_scaled": float(np.mean((yv - pred)**2)**.5),
                       "seconds": time.monotonic() - t})
        logging.info("%s tuning %s/%s %.4f", method, i + 1, len(configurations(method)), trials[-1]["rmse_scaled"])
        save_json(ROOT / "results/configs" / f"{method}_trials.json", trials)
    chosen = min(trials, key=lambda r: r["rmse_scaled"] if r["rmse_scaled"] is not None else 0)
    save_json(path, {"method": method, "params": chosen["params"], "trials": trials,
                     "split": "train targets through 2017-12; tuning targets 2018-01..02"})


def execute_case(method, case, df, prefix_models=False, seeds=SEEDS):
    destination = ROOT / "results/runs" / method / f"{case}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    params = json.loads(config_path(method).read_text())["params"]
    start = time.monotonic()
    x, y, xv, yv, s, sv, pre = transformed(df, method, ("train", "tune"), "rank")
    prefix = ROOT / "models" / method / case
    try:
        result = run_selector(method, x, y, xv, yv, s, params, seeds=seeds,
                              prefix=prefix if prefix_models else None)
        out = {"status": "ok", "method": method, "case": case, "params": params,
               "score": result.score, "order": result.order, "native": result.native,
               "metadata": result.metadata, "fit_row_hash": digest_array(
                   df.loc[df.split.isin(["train", "tune"]), "row_id"].to_numpy()),
               "rank_row_hash": digest_array(df.loc[df.split.eq("rank"), "row_id"].to_numpy()),
               "test_rows_read_by_selector": 0,
               "total_seconds": time.monotonic() - start}
    except Exception as exc:
        out = {"status": "failed", "method": method, "case": case, "error": str(exc),
               "traceback": traceback.format_exc(), "total_seconds": time.monotonic() - start}
        logging.exception("%s %s failed", method, case)
    save_json(destination, out)
    logging.info("%s %s %s %.1fs", method, case, out["status"], out["total_seconds"])
    return out


def run_real(method, part="all", shard=0, shards=1):
    tune(method)
    df = load_data()
    dev = df.loc[df.split.ne("test")].copy()
    ref = execute_case(method, "reference", dev, True)
    if ref["status"] != "ok":
        return
    if part == "reference":
        return
    manifest_path = ROOT / "data_processed/sampling_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else make_sampling_manifest()
    for index, spec in enumerate(manifest):
        if index % shards != shard:
            continue
        case = dev.loc[dev.seller_id.isin(spec["sellers"])]
        execute_case(method, spec["run"], case)
    for batch in range(10):
        if batch % shards != shard:
            continue
        execute_case(method, f"seedonly_{batch:02d}", dev, seeds=[s + 1000 * (batch + 1) for s in SEEDS])


def simulation_data(scenario, repetition):
    path = ROOT / "data_processed/simulations" / f"{scenario['name']}_{repetition:03d}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    seed = 330000 + repetition + 1000 * PROTOCOL["simulation"]["scenarios"].index(scenario)
    rng = np.random.default_rng(seed)
    n, p = 2000, 31
    sigma = .65 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    z = rng.multivariate_normal(np.zeros(p), sigma, n)
    x = z.copy()
    x[:, ::3] = np.exp(.5 * x[:, ::3])
    x[:, -2:] = (x[:, -2:] > .7).astype(float)
    truth = rng.choice(p - 2, scenario["signals"], replace=False)
    beta = rng.choice([-1., 1.], len(truth)) * rng.uniform(.7, 1.3, len(truth))
    standard = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    if scenario["response"] == "linear":
        signal = standard[:, truth] @ beta
    else:
        active = standard[:, truth]
        signal = np.sin(active[:, :5] * 1.8) @ beta[:5]
        signal += (active[:, 5:] ** 2 - 1) @ beta[5:]
        signal += active[:, 0] * active[:, 5] + active[:, 1] * active[:, 6]
    signal /= max(signal.std(), 1e-6)
    groups = np.arange(n)
    noise = rng.normal(size=n)
    if scenario["response"] == "nonlinear_panel":
        groups = np.repeat(np.arange(n // 10), 10)
        noise = (noise + np.repeat(rng.normal(size=n // 10), 10)) / np.sqrt(2)
    y = signal + noise / np.sqrt(scenario["snr"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), y=y, truth=truth, groups=groups)
    return dict(x=x.astype(np.float32), y=y, truth=truth, groups=groups)


def run_simulations(method, shard=0, shards=1):
    tune(method)
    params = json.loads(config_path(method).read_text())["params"]
    for scenario in PROTOCOL["simulation"]["scenarios"]:
        for rep in range(100):
            if rep % shards != shard:
                continue
            path = ROOT / "results/simulation_runs" / method / f"{scenario['name']}_{rep:03d}.json"
            if path.exists():
                continue
            data = simulation_data(scenario, rep)
            x, y = data["x"], data["y"]
            pre = Preprocessor(copula=method == "copula_mvr", business=False).fit(x[:1200])
            xt, xv = pre.transform(x[:1200]), pre.transform(x[1200:1600])
            start = time.monotonic()
            try:
                yt, yv = y[:1200], y[1200:1600]
                response_scaling = "raw; scale-equivariant tree loss or normalized subset utility"
                if method in ("elastic_net", "stability_selection"):
                    mean, std = yt.mean(), max(yt.std(), 1e-6)
                    yt, yv = (yt - mean) / std, (yv - mean) / std
                    response_scaling = "training mean and population standard deviation"
                elif method in ("copula_mvr", "deep_lasso", "tabm", "deepdrk"):
                    response_scaling = "training standardization inside selector"
                fitted = run_selector(method, xt, yt, xv, yv,
                                      data["groups"][:1200], params, SEEDS)
                truth = set(map(int, data["truth"]))
                results = []
                sets = [(f"top{k}", fitted.order[:k]) for k in KS]
                if fitted.native is not None:
                    sets.append(("native", fitted.native))
                for name, found in sets:
                    found = set(map(int, found))
                    false = len(found - truth)
                    fdp = false / max(1, len(found))
                    results.append({"rule": name, "FDP": fdp, "power": len(found & truth) / len(truth),
                                    "discoveries": len(found), "FDP_gt_0.2": fdp > .2})
                columns = fitted.order[:12]
                evaluator = xgb_model({"trees": 140, "depth": 3, "lr": .05}, 11).fit(
                    xt[:, columns], y[:1200])
                prediction = evaluator.predict(pre.transform(x[1600:])[:, columns])
                output = {"status": "ok", "method": method, "scenario": scenario, "rep": rep,
                          "data_hash": digest_array(x), "truth": data["truth"], "score": fitted.score,
                          "response_hash": digest_array(y), "response_scaling": response_scaling,
                          "order": fitted.order, "metrics": results,
                          "rmse_at_12": float(np.mean((prediction - y[1600:]) ** 2) ** .5),
                          "seconds": time.monotonic() - start}
            except Exception as exc:
                output = {"status": "failed", "method": method, "scenario": scenario["name"],
                          "rep": rep, "error": str(exc), "traceback": traceback.format_exc()}
            save_json(path, output)
            logging.info("simulation %s %s %03d %s", method, scenario["name"], rep, output["status"])


def evaluate(method):
    if not (ROOT / "results/development_choice.json").exists():
        raise RuntimeError("Freeze development-only model choice before evaluating test targets")
    ref_path = ROOT / "results/runs" / method / "reference.json"
    if not ref_path.exists():
        return
    ref = json.loads(ref_path.read_text())
    if ref["status"] != "ok":
        return
    df = load_data()
    tr = df.loc[df.split.isin(["train", "tune"])]
    va = df.loc[df.split.eq("rank")]
    te = df.loc[df.split.eq("test")]
    for evaluator, tunemethod in (("xgboost", "xgboost_shap"), ("tabm", "tabm")):
        config = json.loads(config_path(tunemethod).read_text())["params"]
        for k in KS + [31]:
            if k == 31 and method != "copula_mvr":
                continue
            name = "all_features" if k == 31 else method
            dest = ROOT / "results/predictions" / f"{name}_{evaluator}_k{k}.csv"
            if dest.exists() and dest.with_suffix(".json").exists():
                continue
            columns = ref["order"][:k] if k < 31 else list(range(31))
            pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
            x = pre.transform(tr[FEATURES].to_numpy(float))[:, columns]
            xv = pre.transform(va[FEATURES].to_numpy(float))[:, columns]
            xtest = pre.transform(te[FEATURES].to_numpy(float))[:, columns]
            y, yv = tr.log_gmv_next_month.to_numpy(), va.log_gmv_next_month.to_numpy()
            valid_preds, preds = [], []
            for seed in SEEDS:
                if evaluator == "xgboost":
                    model = xgb_model(config, seed).fit(x, y)
                    valid_preds.append(model.predict(xv))
                    # Fixed tree count, refit on all development targets.
                    model = xgb_model(config, seed).fit(np.concatenate([x, xv]), np.r_[y, yv])
                    preds.append(model.predict(xtest))
                else:
                    model, meta = fit_neural(x, y, xv, yv, config, "tabm", seed)
                    valid_preds.append(neural_predict(model, xv))
                    best_epoch = min(meta["history"], key=lambda r: r[2])[0]
                    refit_config = config | {"epochs": best_epoch, "fixed_epochs": True}
                    model, _ = fit_neural(np.concatenate([x, xv]), np.r_[y, yv],
                                          xv, yv, refit_config, "tabm", seed)
                    preds.append(neural_predict(model, xtest))
            pred, val = np.mean(preds, 0), np.mean(valid_preds, 0)
            dest.parent.mkdir(parents=True, exist_ok=True)
            out = te[["row_id", "seller_id", "target_month", "log_gmv_next_month"]].copy()
            out["prediction"] = pred
            out.to_csv(dest, index=False)
            save_json(dest.with_suffix(".json"), {
                "method": name, "evaluator": evaluator, "k": k, "features": [FEATURES[j] for j in columns],
                "test": metrics(te.log_gmv_next_month.to_numpy(), pred), "validation": metrics(yv, val),
                "seeds": SEEDS, "test_used_for_training": False})
            logging.info("prediction %s %s K=%s", name, evaluator, k)


def manifest():
    import importlib.metadata
    sources = {}
    for name in ("DeepDRK", "deep_lasso", "tabm"):
        sources[name] = subprocess.check_output(
            ["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True).strip()
    sources["VTFS_zip_sha256"] = digest_file(ROOT / "vendor/VTFS.zip")
    save_json(ROOT / "results/source_manifest.json", {
        "protocol_sha256": digest_file(ROOT / "protocol.json"),
        "vendor_versions": sources,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    })
    make_sampling_manifest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["manifest", "tune", "reference", "stability", "simulate", "evaluate"])
    parser.add_argument("--method", choices=METHODS, default="elastic_net")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "logs" / f"{args.method}_{args.stage}.log")])
    if args.stage == "manifest":
        manifest()
    elif args.stage == "tune":
        tune(args.method)
    elif args.stage == "reference":
        run_real(args.method, "reference")
    elif args.stage == "stability":
        run_real(args.method, shard=args.shard, shards=args.shards)
    elif args.stage == "simulate":
        run_simulations(args.method, args.shard, args.shards)
    elif args.stage == "evaluate":
        evaluate(args.method)


if __name__ == "__main__":
    main()
