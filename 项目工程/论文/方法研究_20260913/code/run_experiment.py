"""New-seed evaluation: no V7 simulation outcome is reused."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
import traceback

from paired_loss import (ROOT, V7, P, SEEDS, execute_methods, gaussian_knockoff,
                         TrainingCopula, loss_family, contexts_for, grouped_statistics,
                         threshold)
from common import (FEATURES, Preprocessor, load_data, digest_array, digest_file,
                    save_json, rank, seed_all, metrics)
from models import xgb_model, fit_neural, neural_predict
import numpy as np
from scipy.stats import t
from sklearn.covariance import LedoitWolf
import pandas as pd
import torch

RESULT = ROOT / "results"


def read(path):
    return json.loads(path.read_text())


def init():
    import importlib.metadata
    if (RESULT / "initial_manifest.json").exists():
        assert read(RESULT / "initial_manifest.json")["protocol_sha"] == digest_file(ROOT / "protocol.json")
        print("Retaining original initial manifest", flush=True)
        return
    save_json(RESULT / "initial_manifest.json", {
        "protocol_sha": digest_file(ROOT / "protocol.json"),
        "panel_sha": digest_file(V7 / "data_processed/seller_month_asof.parquet"),
        "sampling_sha": digest_file(V7 / "data_processed/sampling_manifest.json"),
        "code_sha": {p.name: digest_file(p) for p in (ROOT / "code").glob("*.py")},
        "predictor_parent_sha": digest_file(V7 / "code/models.py"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "initial_time": time.time()})


def new_simulation(scene, rep):
    path = ROOT / "data" / f"{scene['name']}_{rep:03d}.npz"
    if path.exists():
        return dict(np.load(path))
    seed = P["simulation"]["new_seed_base"] + 10000 * P["simulation"]["scenarios"].index(scene) + rep
    rng = np.random.default_rng(seed)
    n, p = 2000, 31
    covariance = scene["rho"] ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    z = rng.multivariate_normal(np.zeros(p), covariance, n)
    truth = rng.choice(p - 2, scene["signals"], replace=False)
    beta = rng.choice([-1., 1.], len(truth))
    if scene["name"] == "misspecified_t":
        z = z / np.sqrt(rng.chisquare(5, n)[:, None] / 5)
    active = z[:, truth]
    if scene["response"] == "null":
        signal = np.zeros(n)
    elif scene["response"] == "linear":
        signal = active @ beta
    elif scene["response"] == "interactions":
        signal = (active[:, ::2] * active[:, 1::2]) @ beta[::2]
    else:
        signal = np.sin(active[:, :5] * 1.5) @ beta[:5]
        signal += ((active[:, 5:]**2 - 1) / np.sqrt(2)) @ beta[5:]
        signal += .5 * active[:, 0] * active[:, 5]
    signal = (signal - signal[:1200].mean()) / max(signal[:1200].std(), 1e-6)
    noise = rng.normal(size=n)
    y = noise if scene["signals"] == 0 else signal + noise / np.sqrt(scene["snr"])
    cov = LedoitWolf().fit(z[:1200]).covariance_ if scene["name"] == "misspecified_t" else covariance
    mu = z[:1200].mean(0) if scene["name"] == "misspecified_t" else None
    zk, _ = gaussian_knockoff(z, cov, seed + 5500000, mu)
    x, xk = z.copy(), zk.copy()
    if scene["name"] == "mixed_copula":
        x[:, :15] = np.exp(.5 * x[:, :15])
        xk[:, :15] = np.exp(.5 * xk[:, :15])
        x[:, -2:] = (x[:, -2:] > .5).astype(float)
        xk[:, -2:] = (xk[:, -2:] > .5).astype(float)
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, x=x, xk=xk, y=y, truth=truth, covariance=cov,
                        seed=seed, groups=np.arange(n))
    return dict(x=x, xk=xk, y=y, truth=truth, groups=np.arange(n))


def discovery_metrics(found, truth):
    found, truth = set(map(int, found)), set(map(int, truth))
    return {"FDP": len(found - truth) / max(1, len(found)),
            "power": len(found & truth) / len(truth) if truth else None,
            "size": len(found), "false_count": len(found - truth)}


def downstream(x, y, xt, yt, selected, cache):
    key = tuple(sorted(selected))
    if key not in cache:
        if not key:
            prediction = np.full(len(yt), y.mean())
        else:
            model = xgb_model({"trees": 140, "depth": 3, "lr": .05}, 11).fit(x[:, key], y)
            prediction = model.predict(xt[:, key])
        cache[key] = float(np.mean((prediction - yt)**2)**.5)
    return cache[key]


def simulations(shard, shards):
    for i, (scene, rep) in enumerate((s, r) for s in P["simulation"]["scenarios"] for r in range(100)):
        if i % shards != shard:
            continue
        case = f"{scene['name']}_{rep:03d}"
        destination = RESULT / "simulations" / f"{case}.json"
        if destination.exists():
            continue
        start = time.monotonic()
        raw = new_simulation(scene, rep)
        try:
            pre = Preprocessor(business=False).fit(raw["x"][:1200])
            x, xk = pre.transform(raw["x"][:1600]), pre.transform(raw["xk"][:1600])
            out, meta = execute_methods(x, raw["y"][:1600], xk, 1200, 1000, raw["groups"][:1600])
            assert set(out) == set(P["methods"])
            cache = {}
            for name, record in out.items():
                record["top12"] = discovery_metrics(record["order"][:12], raw["truth"])
                record["top8"] = discovery_metrics(record["order"][:8], raw["truth"])
                record["rmse12"] = downstream(x, raw["y"][:1600], pre.transform(raw["x"][1600:]),
                                             raw["y"][1600:], record["order"][:12], cache)
                if record["native"] is not None:
                    record["native_metrics"] = discovery_metrics(record["native"], raw["truth"])
                    record["native10_metrics"] = discovery_metrics(record["native_q10"], raw["truth"])
                    record["rmse_native"] = downstream(x, raw["y"][:1600], pre.transform(raw["x"][1600:]),
                                                       raw["y"][1600:], record["native"], cache)
            save_json(destination, {"status": "ok", "scene": scene, "rep": rep, "methods": out,
                "training": meta, "x_hash": digest_array(raw["x"]), "y_hash": digest_array(raw["y"]),
                "xk_hash": digest_array(raw["xk"]), "truth": raw["truth"],
                "protocol_sha": digest_file(ROOT / "protocol.json"), "seconds": time.monotonic() - start})
        except Exception as exc:
            save_json(destination, {"status": "failed", "case": case, "error": str(exc),
                                    "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
        print(case, "seconds", round(time.monotonic() - start, 2), flush=True)


def real(shard, shards, mvr=False):
    frame = load_data().loc[lambda d: d.split.ne("test")]
    cases = [{"run": "reference", "sellers": None, "seed_offset": 0}]
    cases += [s | {"seed_offset": 0} for s in read(V7 / "data_processed/sampling_manifest.json")]
    cases += [{"run": f"seedonly_{i:02d}", "sellers": None, "seed_offset": (i + 1) * 1000}
              for i in range(10)]
    for i, case in enumerate(cases):
        if i % shards != shard:
            continue
        destination = RESULT / ("real_mvr" if mvr else "real") / f"{case['run']}.json"
        if destination.exists():
            continue
        begin = time.monotonic()
        subset = frame if case["sellers"] is None else frame.loc[frame.seller_id.isin(case["sellers"])]
        fit = subset.loc[subset.split.eq("train")]
        early = subset.loc[subset.split.eq("tune")]
        infer = subset.loc[subset.split.eq("rank")]
        stacked = pd.concat([fit, early, infer])
        ntrain = len(fit) + len(early)
        seeds = [s + case["seed_offset"] for s in SEEDS]
        try:
            pre = Preprocessor().fit(stacked[FEATURES].to_numpy(float)[:ntrain])
            x = pre.transform(stacked[FEATURES].to_numpy(float))
            generator_model = TrainingCopula().fit(x[:ntrain])
            if mvr:
                xk_infer, generator = generator_model.generate_mvr(
                    x[ntrain:], P["real_data"]["generator_seed"] + case["seed_offset"])
                xk = np.concatenate([x[:ntrain], xk_infer])
            else:
                xk, generator = generator_model.generate(
                    x, P["real_data"]["generator_seed"] + case["seed_offset"])
            y = stacked.log_gmv_next_month.to_numpy(float)
            groups = stacked.seller_id.to_numpy()
            prefix = ROOT / "models" / "reference" if case["run"] == "reference" and not mvr else None
            out, meta = execute_methods(x, y, xk, ntrain, len(fit), groups, seeds, prefix)
            if mvr:
                original = read(RESULT / "real" / f"{case['run']}.json")
                for name in ("xgb_shap", "xgb_pfi", "tabm_pfi", "deep_lasso"):
                    assert np.allclose(out[name]["score"], original["methods"][name]["score"], rtol=1e-6, atol=1e-7)
                generator["extension_protocol_sha"] = digest_file(ROOT / "mvr_extension_protocol.json")
            if case["run"] == "reference":
                filename = "real_generator_pairs_mvr.npz" if mvr else "real_generator_pairs.npz"
                np.savez_compressed(ROOT / "data" / filename, x=x[ntrain:], xk=xk[ntrain:],
                                    row_id=infer.row_id.to_numpy(), y=y[ntrain:])
            save_json(destination, {"status": "ok", "case": case["run"], "methods": out,
                "training": meta, "seeds": seeds, "generator": generator,
                "fit_row_hash": digest_array(stacked.row_id.to_numpy()[:ntrain]),
                "inference_row_hash": digest_array(infer.row_id.to_numpy()),
                "test_rows_read_by_selector": 0, "inference_labels_used_for_training": False,
                "protocol_sha": digest_file(ROOT / "protocol.json"), "seconds": time.monotonic() - begin})
        except Exception as exc:
            save_json(destination, {"status": "failed", "case": case["run"], "error": str(exc),
                                    "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
        print(case["run"], "seconds", round(time.monotonic() - begin, 2), flush=True)


def algebra_checks():
    seed_all(11)
    rng = np.random.default_rng(381)
    x, xk = rng.normal(size=(128, 8)), rng.normal(size=(128, 8))
    y = rng.normal(size=128)
    groups = np.repeat(np.arange(32), 4)
    def model(z):
        return np.sin(z[:, 0]) + z[:, 1] * z[:, 2] + .2 * z[:, 3]**2 + z[:, 4:].sum(1)
    base = loss_family(model, x, xk, y, groups)
    rows = []
    for chosen in [[i] for i in range(8)] + [[0, 2, 4], list(range(8))]:
        a, b = x.copy(), xk.copy()
        a[:, chosen], b[:, chosen] = xk[:, chosen], x[:, chosen]
        flipped = loss_family(model, a, b, y, groups)
        for name in base:
            target = np.array(base[name]["score"])
            target[chosen] *= -1
            error = float(np.abs(np.asarray(flipped[name]["score"]) - target).max())
            rows.append({"statistic": name, "swapped": chosen, "max_error": error})
            if name != "naive":
                assert error < 1e-9
    assert max(r["max_error"] for r in rows if r["statistic"] == "naive") > .01
    cov = .6 ** np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    z = rng.multivariate_normal(np.zeros(5), cov, 100000)
    zk, meta = gaussian_knockoff(z, cov, 11)
    target = np.block([[cov, cov - meta["s"] * np.eye(5)],
                       [cov - meta["s"] * np.eye(5), cov]])
    err = float(np.max(np.abs(np.cov(np.column_stack([z, zk]).T) - target)))
    assert err < .03
    # Standard Knockoff+ cannot select fewer than ceil(1/q) variables.
    assert np.isinf(threshold(np.array([1., 1., 1., 1., 1., 0., 0.]), .1))
    assert threshold(np.array([1., 1., 1., 1., 1., 0., 0.]), .2) == 1.
    save_json(RESULT / "algebra_checks.json", {"sign_flip_cases": rows, "oracle_covariance_error": err,
        "knockoff_threshold_minimum_count": "passed", "status": "passed",
        "not_a_proof_of_Olist_exchangeability": True})
    print("Algebra and sampler checks passed", flush=True)


def predict(evaluator, mvr=False):
    ref = read(RESULT / ("real_mvr" if mvr else "real") / "reference.json")
    frame = load_data()
    tr = frame.loc[frame.split.isin(["train", "tune"])]
    va, te = frame.loc[frame.split.eq("rank")], frame.loc[frame.split.eq("test")]
    pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
    allx, allv, allt = [pre.transform(df[FEATURES].to_numpy(float)) for df in (tr, va, te)]
    y, yv = tr.log_gmv_next_month.to_numpy(), va.log_gmv_next_month.to_numpy()
    config = read(V7 / f"results/configs/{'xgboost_shap' if evaluator == 'xgboost' else 'tabm'}.json")["params"]
    methods = ref["methods"] | {"all_features": {"order": list(range(31)), "native": None}}
    for method, record in methods.items():
        sets = {"top12": record["order"][:12], "top8": record["order"][:8]}
        if mvr:
            sets = {"top12": record["order"][:12]}
        if method == "all_features":
            sets = {"all": record["order"]}
        if record["native"] is not None and not mvr:
            sets["native"] = record["native"]
        for rule, columns in sets.items():
            columns = sorted(columns)
            destination = RESULT / ("predictions_mvr" if mvr else "predictions") / f"{method}_{evaluator}_{rule}.csv"
            if destination.exists() and destination.with_suffix(".json").exists():
                continue
            key = "_".join(map(str, columns)) + "_" + evaluator
            cachefile = RESULT / "prediction_cache" / f"{key}.npz"
            if cachefile.exists():
                arrays = np.load(cachefile)
                pred, val = arrays["pred"], arrays["val"]
            elif not columns:
                pred, val = np.full(len(te), np.r_[y, yv].mean()), np.full(len(va), y.mean())
            else:
                x, xv, xt = allx[:, columns], allv[:, columns], allt[:, columns]
                vals, preds = [], []
                for seed in SEEDS:
                    if evaluator == "xgboost":
                        model = xgb_model(config, seed).fit(x, y)
                        vals.append(model.predict(xv))
                        model = xgb_model(config, seed).fit(np.concatenate([x, xv]), np.r_[y, yv])
                        preds.append(model.predict(xt))
                    else:
                        model, meta = fit_neural(x, y, xv, yv, config, "tabm", seed)
                        vals.append(neural_predict(model, xv))
                        epoch = min(meta["history"], key=lambda r: r[2])[0]
                        model, _ = fit_neural(np.concatenate([x, xv]), np.r_[y, yv], xv, yv,
                                             config | {"epochs": epoch, "fixed_epochs": True,
                                                       "patience": epoch + 1}, "tabm", seed)
                        preds.append(neural_predict(model, xt))
                pred, val = np.mean(preds, 0), np.mean(vals, 0)
                cachefile.parent.mkdir(exist_ok=True)
                np.savez_compressed(cachefile, pred=pred, val=val)
            out = te[["row_id", "seller_id", "target_month", "log_gmv_next_month"]].copy()
            out["prediction"] = pred
            destination.parent.mkdir(exist_ok=True)
            out.to_csv(destination, index=False)
            save_json(destination.with_suffix(".json"), {
                "method": method, "rule": rule, "evaluator": evaluator, "columns": columns,
                "features": [FEATURES[i] for i in columns], "test": metrics(te.log_gmv_next_month, pred),
                "development": metrics(yv, val), "test_previously_observed": True})
            print("prediction", method, evaluator, rule, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["init", "checks", "real", "simulate", "full", "predict",
                                         "mvr-real", "mvr-predict"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--evaluator", choices=["xgboost", "tabm"], default="xgboost")
    args = parser.parse_args()
    for directory in ("results", "data", "models", "logs"):
        (ROOT / directory).mkdir(exist_ok=True)
    logfile = ROOT / "logs" / f"{args.stage}_{args.evaluator}_{args.shard}.log"
    with logfile.open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        if args.stage == "init":
            init()
        elif args.stage == "checks":
            algebra_checks()
        elif args.stage == "predict":
            predict(args.evaluator)
        elif args.stage == "mvr-real":
            real(args.shard, args.shards, mvr=True)
        elif args.stage == "mvr-predict":
            predict(args.evaluator, mvr=True)
        else:
            if args.stage in ("real", "full"):
                real(args.shard, args.shards)
            if args.stage in ("simulate", "full"):
                simulations(args.shard, args.shards)
    print("Finished", args.stage, args.shard, flush=True)
