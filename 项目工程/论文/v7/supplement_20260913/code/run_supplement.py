"""Published Semi-KO mechanism and conditional-moment ablations."""

from pathlib import Path
import argparse
import ast
import contextlib
import importlib.util
import json
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT.parent
SCPL = V7.parent / "方法研究_20260913"
sys.path.insert(0, str(SCPL / "code"))
from paired_loss import selection, threshold, SEEDS, permutation_score
from common import Preprocessor, FEATURES, load_data, digest_array, digest_file, save_json, metrics
from models import xgb_model, fit_neural, neural_predict
from run_experiment import downstream, discovery_metrics
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

P = json.loads((ROOT / "protocol.json").read_text())
MODES = ("ridge", "quad", "scale", "quad_scale")
RESULT = ROOT / "results"


def read(path):
    return json.loads(path.read_text())


def basis(x, y=None):
    columns = [x, x**2]
    if y is not None:
        columns += [y[:, None], (y**2)[:, None], x * y[:, None]]
    return np.column_stack(columns)


def ridge_prediction(design, target, scaled=False):
    if scaled:
        design = StandardScaler().fit_transform(design)
    return Ridge(alpha=1.0, solver="cholesky").fit(design, target).predict(design)


def perturbations(x, y, seed):
    n, p = x.shape
    rng = np.random.RandomState(seed)
    permutations = [[rng.permutation(n) for _ in range(p)] for _ in range(2)]
    perturbed = {mode: [[], []] for mode in MODES}
    diagnostic = []
    for j in range(p):
        other = np.delete(x.astype(float), j, axis=1)
        target = x[:, j].astype(float)
        designs = [other, np.column_stack([other, y])]
        quadratic = [basis(other), basis(other, y)]
        means = {
            "ridge": [ridge_prediction(d, target) for d in designs],
            "quad": [ridge_prediction(d, target, scaled=True) for d in quadratic],
        }
        for mean_mode in ("ridge", "quad"):
            for side in range(2):
                mean = means[mean_mode][side]
                residual = target - mean
                variance = max(float(np.mean(residual**2)), 1e-6)
                log_square = np.log(residual**2 + .05 * variance)
                log_prediction = ridge_prediction(quadratic[side], log_square, scaled=True)
                scale = np.sqrt(np.exp(np.clip(log_prediction, np.log(.05 * variance), np.log(20 * variance))))
                for use_scale in (False, True):
                    mode = ("scale" if mean_mode == "ridge" else "quad_scale") if use_scale else mean_mode
                    values = (mean + scale * (residual / scale)[permutations[side][j]]
                              if use_scale else mean + residual[permutations[side][j]])
                    copy = x.copy()
                    copy[:, j] = values
                    assert np.isfinite(copy).all()
                    perturbed[mode][side].append(copy)
            diagnostic.append({"feature": j, "mean_model": mean_mode,
                               "conditional_mean_gap": float(np.mean((means[mean_mode][0] - means[mean_mode][1])**2))})
    return perturbed, diagnostic


def score_perturbations(predictors, x, y, seed):
    perturbed, diagnostics = perturbations(x, y, seed)
    out = {}
    for model_name, predict in predictors.items():
        for mode in MODES:
            predictions = []
            for side in range(2):
                batch = np.concatenate(perturbed[mode][side])
                predictions.append(predict(batch).reshape(x.shape[1], len(x)))
            score = ((predictions[0] - y)**2 - (predictions[1] - y)**2).mean(axis=1)
            out[f"{model_name}_sko_{mode}"] = selection(score, meta={
                "oracle_guarantee_not_transferred_to_estimated_nuisance": True,
                "nuisance_uses_scoring_response": True, "permutations_per_side": 1})
    return out, diagnostics


def predictors(x, y, nfit, seeds, parent, save=False):
    tx = read(V7 / "results/configs/xgboost_shap.json")["params"]
    tn = read(V7 / "results/configs/tabm.json")["params"]
    trees = [xgb_model(tx, seed).fit(x, y) for seed in seeds]
    networks, metadata = [], []
    for seed, expected in zip(seeds, parent["training"]["tabm"]):
        first, meta = fit_neural(x[:nfit], y[:nfit], x[nfit:], y[nfit:], tn, "tabm", seed)
        epoch = min(meta["history"], key=lambda r: r[2])[0]
        assert epoch == expected["selected_epoch"]
        model, history = fit_neural(x, y, x[nfit:], y[nfit:],
                                    tn | {"epochs": epoch, "fixed_epochs": True, "patience": epoch + 1},
                                    "tabm", seed)
        networks.append(model)
        metadata.append({"seed": seed, "epoch": epoch, "refit_epochs": history["epochs"]})
        if save:
            (ROOT / "models").mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": tn,
                        "y_mean": model.y_mean, "y_std": model.y_std}, ROOT / "models" / f"tabm_{seed}.pt")
    return {
        "xgb": lambda z: np.mean([m.predict(z) for m in trees], axis=0),
        "tabm": lambda z: np.mean([neural_predict(m, z) for m in networks], axis=0),
    }, metadata


def real(shard, shards):
    frame = load_data().loc[lambda d: d.split.ne("test")]
    cases = [{"run": "reference", "sellers": None, "seed_offset": 0}]
    cases += [s | {"seed_offset": 0} for s in read(V7 / "data_processed/sampling_manifest.json")]
    cases += [{"run": f"seedonly_{i:02d}", "sellers": None, "seed_offset": (i + 1) * 1000} for i in range(10)]
    for i, case in enumerate(cases):
        destination = RESULT / "real" / f"{case['run']}.json"
        if i % shards != shard or destination.exists():
            continue
        start = time.monotonic()
        try:
            parent = read(SCPL / "results/real" / destination.name)
            d = frame if case["sellers"] is None else frame.loc[frame.seller_id.isin(case["sellers"])]
            train, early, infer = [d.loc[d.split.eq(s)] for s in ("train", "tune", "rank")]
            tr = pd.concat([train, early])
            pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
            x, xi = pre.transform(tr[FEATURES].to_numpy(float)), pre.transform(infer[FEATURES].to_numpy(float))
            y, yi = tr.log_gmv_next_month.to_numpy(), infer.log_gmv_next_month.to_numpy()
            seeds = [s + case["seed_offset"] for s in SEEDS]
            pred, meta = predictors(x, y, len(train), seeds, parent, case["run"] == "reference")
            for model in ("xgb", "tabm"):
                check = permutation_score(pred[model], xi, yi, seeds[0])
                assert np.allclose(check, parent["methods"][model + "_pfi"]["score"], rtol=2e-6, atol=1e-6)
            out, diagnostic = score_perturbations(pred, xi, yi, seeds[0] + 9921)
            save_json(destination, {
                "status": "ok", "case": case["run"], "methods": out, "training": meta,
                "fit_row_hash": digest_array(tr.row_id.to_numpy()),
                "inference_row_hash": digest_array(infer.row_id.to_numpy()),
                "parent_scores_matched": True, "nuisance_uses_rank_labels": True,
                "test_rows_used_for_selection": 0, "diagnostic": diagnostic,
                "protocol_sha": digest_file(ROOT / "protocol.json"), "seconds": time.monotonic() - start})
        except Exception as exc:
            save_json(destination, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
        print(case["run"], round(time.monotonic() - start, 2), flush=True)


def simulate(shard, shards):
    for i, path in enumerate(sorted((SCPL / "results/simulations").glob("*.json"))):
        destination = RESULT / "simulations" / path.name
        if i % shards != shard or destination.exists():
            continue
        start = time.monotonic()
        try:
            parent = read(path)
            raw = dict(np.load(SCPL / "data" / path.with_suffix(".npz").name))
            pre = Preprocessor(business=False).fit(raw["x"][:1200])
            x, xt = pre.transform(raw["x"][:1600]), pre.transform(raw["x"][1600:])
            y = raw["y"]
            pred, meta = predictors(x[:1200], y[:1200], 1000, SEEDS, parent)
            out, diagnostic = score_perturbations(pred, x[1200:], y[1200:1600], SEEDS[0] + 9921)
            cache = {}
            for record in out.values():
                for rule, found in (("top12", record["order"][:12]), ("top8", record["order"][:8]),
                                    ("native", record["native"]), ("native10", record["native_q10"])):
                    record[rule + "_metrics"] = discovery_metrics(found, raw["truth"])
                record["rmse12"] = downstream(x, y[:1600], xt, y[1600:], record["order"][:12], cache)
                record["rmse_native"] = downstream(x, y[:1600], xt, y[1600:], record["native"], cache)
            save_json(destination, {
                "status": "ok", "scene": parent["scene"], "rep": parent["rep"], "methods": out,
                "x_hash": digest_array(raw["x"]), "y_hash": digest_array(raw["y"]),
                "parent_file": str(path), "training": meta, "diagnostic": diagnostic,
                "protocol_sha": digest_file(ROOT / "protocol.json"), "seconds": time.monotonic() - start})
        except Exception as exc:
            save_json(destination, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
        print(path.stem, round(time.monotonic() - start, 2), flush=True)


def predict(evaluator):
    ref = read(RESULT / "real/reference.json")["methods"]
    frame = load_data()
    tr, va, te = [frame.loc[frame.split.isin(s)] for s in (["train", "tune"], ["rank"], ["test"])]
    pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
    x, xv, xt = [pre.transform(d[FEATURES].to_numpy(float)) for d in (tr, va, te)]
    y, yv = tr.log_gmv_next_month.to_numpy(), va.log_gmv_next_month.to_numpy()
    config = read(V7 / f"results/configs/{'xgboost_shap' if evaluator == 'xgboost' else 'tabm'}.json")["params"]
    for method, record in ref.items():
        for k in (12, 8):
            dest = RESULT / "predictions" / f"{method}_{evaluator}_top{k}.csv"
            if dest.exists() and dest.with_suffix(".json").exists():
                continue
            cols = sorted(record["order"][:k])
            key = "_".join(map(str, cols)) + "_" + evaluator + ".npz"
            existing = SCPL / "results/prediction_cache" / key
            cache = RESULT / "prediction_cache" / key
            if existing.exists() or cache.exists():
                arrays = np.load(existing if existing.exists() else cache)
                pred, val = arrays["pred"], arrays["val"]
            else:
                vals, preds = [], []
                for seed in SEEDS:
                    if evaluator == "xgboost":
                        model = xgb_model(config, seed).fit(x[:, cols], y)
                        vals.append(model.predict(xv[:, cols]))
                        model = xgb_model(config, seed).fit(np.concatenate([x[:, cols], xv[:, cols]]), np.r_[y, yv])
                        preds.append(model.predict(xt[:, cols]))
                    else:
                        model, meta = fit_neural(x[:, cols], y, xv[:, cols], yv, config, "tabm", seed)
                        vals.append(neural_predict(model, xv[:, cols]))
                        epoch = min(meta["history"], key=lambda r: r[2])[0]
                        model, _ = fit_neural(np.concatenate([x[:, cols], xv[:, cols]]), np.r_[y, yv],
                                             xv[:, cols], yv, config | {"epochs": epoch, "fixed_epochs": True,
                                                                       "patience": epoch + 1}, "tabm", seed)
                        preds.append(neural_predict(model, xt[:, cols]))
                pred, val = np.mean(preds, 0), np.mean(vals, 0)
                cache.parent.mkdir(exist_ok=True)
                np.savez_compressed(cache, pred=pred, val=val)
            result = te[["row_id", "seller_id", "target_month", "log_gmv_next_month"]].copy()
            result["prediction"] = pred
            dest.parent.mkdir(exist_ok=True)
            result.to_csv(dest, index=False)
            save_json(dest.with_suffix(".json"), {"method": method, "evaluator": evaluator, "rule": f"top{k}",
                "columns": cols, "test": metrics(te.log_gmv_next_month, pred), "development": metrics(yv, val)})
            print(dest.stem, flush=True)


def checks():
    original_utils = ast.parse((ROOT / "references/semi_utils.py").read_text())
    function = next(n for n in original_utils.body if isinstance(n, ast.FunctionDef) and n.name == "knockoff_threshold")
    shim = types.ModuleType("utils")
    shim.np = np
    exec(compile(ast.Module(body=[function], type_ignores=[]), "author_threshold", "exec"), shim.__dict__)
    previous = sys.modules.get("utils")
    sys.modules["utils"] = shim
    spec = importlib.util.spec_from_file_location("author_semi", ROOT / "references/semi_KO.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if previous is None:
        del sys.modules["utils"]
    else:
        sys.modules["utils"] = previous
    rng = np.random.default_rng(187)
    x, y = rng.normal(size=(48, 5)), rng.normal(size=48)
    model = Ridge().fit(x, y)
    original = module.Semi_KO(model, Ridge(alpha=1., solver="cholesky"),
                              Ridge(alpha=1., solver="cholesky"), random_state=19, n_jobs=1)
    original.fit(x, y)
    expected = original.score(x, y, p_val="FDR", n_perm=1, fdr=.2)
    batch, _ = perturbations(x, y, 19)
    predictions = [model.predict(np.concatenate(side)).reshape(5, len(x)) for side in batch["ridge"]]
    observed = ((predictions[0] - y)**2 - (predictions[1] - y)**2).mean(1)
    error = float(np.abs(expected["importance"] - observed).max())
    assert error < 1e-12
    no_solution = np.array([1., 2., 3., 4.])
    author_threshold = float(shim.knockoff_threshold(no_solution, fdr=.2))
    assert author_threshold == 4.
    assert not selection(no_solution)["native"]
    save_json(RESULT / "author_equivalence.json", {"passed": True, "max_score_error": error,
        "raw_source_sha": digest_file(ROOT / "references/semi_KO.py"),
        "source_commit": P["source_commit"], "threshold_source": "author AST function without unrelated imports",
        "scope": "Ridge fit/predict, permutation order, vectorized score; native threshold corrected to paper equation",
        "author_threshold_no_solution_bug": {"W": no_solution, "q": .2,
            "author_threshold": author_threshold, "author_selected": [3], "correct_selected": []},
        "not_an_algorithmic_innovation": True})
    save_json(RESULT / "initial_manifest.json", {"protocol_sha": digest_file(ROOT / "protocol.json"),
        "panel_sha": digest_file(V7 / "data_processed/seller_month_asof.parquet"),
        "code_sha": digest_file(Path(__file__)), "time": time.time()})
    print("Author-equivalence checks passed", error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["checks", "real", "simulate", "full", "predict"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--evaluator", default="xgboost")
    args = parser.parse_args()
    RESULT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / f"{args.stage}_{args.evaluator}_{args.shard}.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if args.stage == "checks":
                checks()
            elif args.stage == "predict":
                predict(args.evaluator)
            else:
                if args.stage in ("real", "full"):
                    real(args.shard, args.shards)
                if args.stage in ("simulate", "full"):
                    simulate(args.shard, args.shards)
    print("Finished", args.stage, args.shard)
