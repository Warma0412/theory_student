"""Checkpointed full-size paired experiment for four recent model families."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from mechanisms import (STUDY, V7, deep_family, gated_family, entry_family, subset_family,
                        soft_cardinality, jacobian_scores)
from common import (FEATURES, Preprocessor, digest_array, digest_file, load_data,
                    save_json, seed_all, metrics)
from models import xgb_model, fit_neural, neural_predict
import numpy as np
import pandas as pd
import torch

P = json.loads((STUDY / "protocol.json").read_text())
SEEDS = P["tuning"]["neural_seeds"]
FAMILIES = {
    "deep": ["dl_jacobian", "dl_jacobian_lossrank", "dl_score_only"],
    "gate": ["tabm_budget_gate"],
    "entry": ["ep_regression", "ep_refresh", "ep_dispersion"],
    "subset": ["vtfs_cardinality", "vtfs_pairrank", "random_k_search"],
}
METHODS = sum(FAMILIES.values(), [])
RESULT = STUDY / "results"


def read(path):
    return json.loads(Path(path).read_text())


def env_blocks(months):
    unique = np.unique(months)
    mapping = {month: block for block, part in enumerate(np.array_split(unique, min(3, len(unique))))
               for month in part}
    return np.asarray([mapping[m] for m in months])


def prepare(tr, va):
    pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
    y = tr.log_gmv_next_month.to_numpy(float)
    yv = va.log_gmv_next_month.to_numpy(float)
    mean, std = y.mean(), max(y.std(), 1e-6)
    return {"x": pre.transform(tr[FEATURES].to_numpy(float)), "y": (y - mean) / std,
            "xv": pre.transform(va[FEATURES].to_numpy(float)), "yv": (yv - mean) / std,
            "env": env_blocks(tr.target_month.to_numpy()),
            "fit_row_hash": digest_array(tr.row_id.to_numpy()),
            "rank_row_hash": digest_array(va.row_id.to_numpy())}


def family_call(family, data, value, k=12, seeds=SEEDS, prefix=None):
    x, y, xv, yv = [data[key] for key in ("x", "y", "xv", "yv")]
    if prefix:
        prefix.parent.mkdir(parents=True, exist_ok=True)
    if family == "deep":
        params = read(V7 / "results/configs/deep_lasso.json")["params"]
        return deep_family(x, y, xv, yv, params, value, seeds, prefix)
    if family == "gate":
        params = read(V7 / "results/configs/tabm.json")["params"]
        return gated_family(x, y, xv, yv, params, value, k, seeds, prefix)
    if family == "entry":
        return entry_family(x, y, xv, yv, data["env"], k, value, seeds, prefix)
    return subset_family(x, y, xv, yv, k, seeds, prefix)


def initialize():
    import importlib.metadata
    for name in ("logs", "results", "models"):
        (STUDY / name).mkdir(exist_ok=True)
    source = {
        "protocol_hash": digest_file(STUDY / "protocol.json"),
        "panel_hash": digest_file(V7 / "data_processed/seller_month_asof.parquet"),
        "sampling_hash": digest_file(V7 / "data_processed/sampling_manifest.json"),
        "code_hashes": {p.name: digest_file(p) for p in (STUDY / "code").glob("*.py")},
        "parent_code_hashes": {p.name: digest_file(p) for p in (V7 / "code").glob("*.py")},
        "entryprune_commit": subprocess.check_output(
            ["git", "-C", str(V7 / "innovation_benchmark/vendor/EntryPrune"),
             "rev-parse", "HEAD"], text=True).strip(),
        "parent_source_manifest": read(V7 / "results/source_manifest.json"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "created_unix_time": time.time(),
    }
    destination = RESULT / "source_manifest.json"
    if not destination.exists():
        save_json(destination, source)
    else:
        assert read(destination)["protocol_hash"] == source["protocol_hash"], "Protocol changed"
    print("Initialized independent study", flush=True)


def validate_mechanics():
    seed_all(11)
    logits = torch.randn(31, dtype=torch.float64, requires_grad=True)
    for k in (8, 12):
        soft = soft_cardinality(logits, k, .5)
        assert abs(float(soft.sum().detach()) - k) < 1e-6
        grad, = torch.autograd.grad((soft * torch.arange(31)).sum(), logits)
        assert torch.isfinite(grad).all() and abs(float(grad.sum())) < 1e-5
    linear = torch.nn.Linear(31, 1)
    x = np.random.default_rng(0).normal(size=(521, 31)).astype(np.float32)
    score = jacobian_scores(linear, x)
    assert np.allclose(score, linear.weight.detach().numpy()[0].__abs__(), atol=1e-6)
    assert np.allclose(score, jacobian_scores(linear, x[::-1].copy()), atol=1e-6)
    frame = load_data()
    assert len(frame) == 13754 and not frame.row_id.duplicated().any()
    save_json(RESULT / "mechanic_validation.json", {"soft_gate_cardinality": "passed",
        "gate_shift_derivative_zero": "passed", "linear_jacobian_known_answer": "passed",
        "jacobian_batch_and_permutation_invariance": "passed", "panel_keys": "passed"})
    print("Mechanic validation passed", flush=True)


def tune(family):
    destination = RESULT / "configs" / f"{family}.json"
    if destination.exists():
        assert read(destination)["status"] == "ok"
        return
    parameters = {"deep": [.001, .01, .1], "gate": [.001, .01, .05],
                  "entry": [.25, .5, 1.0], "subset": [None]}[family]
    if family == "subset":
        save_json(destination, {"status": "ok", "value": None,
                               "selection": "fixed cardinality and pairwise weight 0.1 in protocol"})
        return
    frame = load_data()
    training = frame.loc[frame.split.eq("train")]
    january = frame.loc[frame.target_month.eq("2018-01-01")]
    february = frame.loc[frame.target_month.eq("2018-02-01")]
    data = prepare(training, january)
    fit = frame.loc[frame.target_month <= "2018-01-01"]
    pre = Preprocessor().fit(fit[FEATURES].to_numpy(float))
    xx, xxv = pre.transform(fit[FEATURES].to_numpy(float)), pre.transform(february[FEATURES].to_numpy(float))
    yy, yyv = fit.log_gmv_next_month.to_numpy(float), february.log_gmv_next_month.to_numpy(float)
    evaluator = read(V7 / "results/configs/xgboost_shap.json")["params"]
    target = {"deep": "dl_jacobian", "gate": "tabm_budget_gate", "entry": "ep_dispersion"}[family]
    trials = []
    for number, value in enumerate(parameters):
        path = RESULT / "configs" / f"{family}_trial{number}.json"
        if path.exists():
            trials.append(read(path))
            continue
        output = family_call(family, data, value)
        cols = output[target]["order"][:12]
        prediction = np.mean([xgb_model(evaluator, seed).fit(xx[:, cols], yy).predict(xxv[:, cols])
                              for seed in SEEDS], 0)
        trial = {"status": "ok", "value": value, "february_rmse": float(np.mean((prediction - yyv)**2)**.5),
                 "order": output[target]["order"], "methods": output, "test_rows_read": 0,
                 "rank_target_months": ["2018-01"], "scoring_target_months": ["2018-02"]}
        save_json(path, trial)
        trials.append(trial)
        print("tune", family, value, trial["february_rmse"], flush=True)
    chosen = min(trials, key=lambda t: (t["february_rmse"], t["value"]))
    save_json(destination, {"status": "ok", "value": chosen["value"], "trials": [
        {k: t[k] for k in ("value", "february_rmse", "order")} for t in trials],
        "test_rows_read": 0, "protocol_hash": digest_file(STUDY / "protocol.json")})


def paths_for(family, case, kind):
    return {m: RESULT / kind / m / f"{case}.json" for m in FAMILIES[family]}


def completed(paths):
    return all(p.exists() for p in paths.values())


def real(family, shard, shards):
    config = read(RESULT / "configs" / f"{family}.json")["value"]
    frame = load_data().loc[lambda d: d.split.ne("test")].copy()
    samples = read(V7 / "data_processed/sampling_manifest.json")
    cases = [{"run": "reference", "sellers": None, "seeds": SEEDS, "k": 12}]
    cases += [s | {"seeds": SEEDS, "k": 12} for s in samples]
    cases += [{"run": f"seedonly_{b:02d}", "sellers": None,
               "seeds": [s + 1000 * (b + 1) for s in SEEDS], "k": 12} for b in range(10)]
    cases += [{"run": "reference_k8", "sellers": None, "seeds": SEEDS, "k": 8}]
    for i, case in enumerate(cases):
        if i % shards != shard:
            continue
        paths = paths_for(family, case["run"], "runs")
        if completed(paths):
            continue
        subset = frame if case["sellers"] is None else frame.loc[frame.seller_id.isin(case["sellers"])]
        data = prepare(subset.loc[subset.split.isin(["train", "tune"])], subset.loc[subset.split.eq("rank")])
        started = time.monotonic()
        try:
            prefix = STUDY / "models" / f"{family}_{case['run']}" if case["run"].startswith("reference") else None
            if family == "deep" and case["k"] == 8 and completed(paths_for(family, "reference", "runs")):
                outputs = {m: read(RESULT / "runs" / m / "reference.json") for m in FAMILIES[family]}
                outputs = {m: {"score": r["score"], "order": r["order"],
                               "metadata": {"same_k_independent_ranking_as_reference": True}} for m, r in outputs.items()}
            else:
                outputs = family_call(family, data, config, case["k"], case["seeds"], prefix)
            for method, output in outputs.items():
                assert sorted(output["order"]) == list(range(31))
                save_json(paths[method], {"status": "ok", "method": method, "case": case["run"], **output,
                    "k": case["k"], "seeds": case["seeds"], "hyperparameter": config,
                    "fit_row_hash": data["fit_row_hash"], "rank_row_hash": data["rank_row_hash"],
                    "test_rows_read_by_selector": 0, "family_elapsed_seconds": time.monotonic() - started,
                    "protocol_hash": digest_file(STUDY / "protocol.json")})
        except Exception as exc:
            for method, path in paths.items():
                if not path.exists():
                    save_json(path, {"status": "failed", "method": method, "case": case["run"],
                                     "error": str(exc), "traceback": traceback.format_exc()})
            logging.exception("Real case failed")
        print(family, case["run"], "finished", round(time.monotonic() - started, 2), flush=True)


def simulate(family, shard, shards):
    config = read(RESULT / "configs" / f"{family}.json")["value"]
    scenarios = read(V7 / "protocol.json")["simulation"]["scenarios"]
    for i, (scenario, rep) in enumerate((s, r) for s in scenarios for r in range(100)):
        if i % shards != shard:
            continue
        case = f"{scenario['name']}_{rep:03d}"
        paths = paths_for(family, case, "simulation_runs")
        if completed(paths):
            continue
        raw = dict(np.load(V7 / "data_processed/simulations" / f"{case}.npz"))
        x, y = raw["x"], raw["y"]
        pre = Preprocessor(business=False).fit(x[:1200])
        xt, xv, xte = pre.transform(x[:1200]), pre.transform(x[1200:1600]), pre.transform(x[1600:])
        # Network functions perform their own training-response standardization.
        data = {"x": xt, "y": y[:1200], "xv": xv, "yv": y[1200:1600],
                "env": np.repeat(np.arange(3), 400)}
        started = time.monotonic()
        try:
            outputs = family_call(family, data, config)
            for method, output in outputs.items():
                cols = output["order"][:12]
                assert len(set(cols)) == 12
                truth = set(map(int, raw["truth"]))
                found = set(map(int, cols))
                evaluator = xgb_model({"trees": 140, "depth": 3, "lr": .05}, 11).fit(xt[:, cols], y[:1200])
                pred = evaluator.predict(xte[:, cols])
                save_json(paths[method], {"status": "ok", "method": method, "scenario": scenario["name"],
                    "rep": rep, "order": output["order"], "score": output["score"],
                    "FDP": len(found - truth) / 12, "power": len(found & truth) / len(truth),
                    "rmse": float(np.mean((pred - y[1600:])**2)**.5),
                    "data_hash": digest_array(x), "response_hash": digest_array(y), "truth": raw["truth"],
                    "metadata": output["metadata"], "family_elapsed_seconds": time.monotonic() - started,
                    "protocol_hash": digest_file(STUDY / "protocol.json")})
        except Exception as exc:
            for method, path in paths.items():
                if not path.exists():
                    save_json(path, {"status": "failed", "method": method, "case": case,
                                     "error": str(exc), "traceback": traceback.format_exc()})
            logging.exception("Simulation failed")
        print(family, case, "finished", round(time.monotonic() - started, 2), flush=True)


def predict_method(method):
    assert all((RESULT / "configs" / f"{family}.json").exists() for family in FAMILIES)
    frame = load_data()
    tr = frame.loc[frame.split.isin(["train", "tune"])]
    va, te = frame.loc[frame.split.eq("rank")], frame.loc[frame.split.eq("test")]
    for k in (12, 8):
        case = "reference" if k == 12 else "reference_k8"
        ref = read(RESULT / "runs" / method / f"{case}.json")
        assert ref["status"] == "ok"
        columns = ref["order"][:k]
        pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
        x, xv, xt = [pre.transform(df[FEATURES].to_numpy(float))[:, columns] for df in (tr, va, te)]
        y, yv = tr.log_gmv_next_month.to_numpy(), va.log_gmv_next_month.to_numpy()
        for evaluator, tune_method in (("xgboost", "xgboost_shap"), ("tabm", "tabm")):
            dest = RESULT / "predictions" / f"{method}_{evaluator}_k{k}.csv"
            if dest.exists() and dest.with_suffix(".json").exists():
                continue
            config = read(V7 / f"results/configs/{tune_method}.json")["params"]
            validations, predictions = [], []
            started = time.monotonic()
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
            out = te[["row_id", "seller_id", "target_month", "log_gmv_next_month"]].copy()
            out["prediction"] = pred
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(dest, index=False)
            month_metrics = {str(m): metrics(d.log_gmv_next_month.to_numpy(), d.prediction.to_numpy())
                             for m, d in out.groupby("target_month")}
            save_json(dest.with_suffix(".json"), {"method": method, "k": k, "evaluator": evaluator,
                "features": [FEATURES[j] for j in columns], "test": metrics(te.log_gmv_next_month.to_numpy(), pred),
                "validation": metrics(yv, val), "test_months": month_metrics, "fit_seeds": SEEDS,
                "test_previously_observed": True, "test_labels_not_used_for_tuning": True,
                "seconds": time.monotonic() - started})
            print("predict", method, evaluator, k, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["init", "validate", "tune", "real", "simulate", "predict", "full"])
    parser.add_argument("--family", choices=list(FAMILIES), default="deep")
    parser.add_argument("--method", choices=METHODS, default="dl_jacobian")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    (STUDY / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    logpath = STUDY / "logs" / f"{args.stage}_{args.family}_{args.method}_{args.shard}.log"
    with logpath.open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        if args.stage == "init":
            initialize()
        elif args.stage == "validate":
            validate_mechanics()
        elif args.stage == "tune":
            tune(args.family)
        elif args.stage == "real":
            real(args.family, args.shard, args.shards)
        elif args.stage == "simulate":
            simulate(args.family, args.shard, args.shards)
        elif args.stage == "full":
            real(args.family, args.shard, args.shards)
            simulate(args.family, args.shard, args.shards)
        else:
            predict_method(args.method)
    print("Finished", args.stage, args.family, args.method, args.shard, flush=True)


if __name__ == "__main__":
    main()
