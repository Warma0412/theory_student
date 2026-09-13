"""Audit shared inputs and report all positive and negative outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from itertools import combinations

from paired_loss import ROOT, V7, P, threshold
from run_experiment import RESULT, read, discovery_metrics
from common import FEATURES, load_data, save_json, digest_array, digest_file, jaccard, metrics, rank
from validate_and_summarize import cinterval, nogueira
import numpy as np
import pandas as pd
from scipy.stats import beta, t

METHODS = P["methods"]
LABELS = {
    "lasso_pair_q20": "评分集成对Lasso",
    "linear_orbit_student": "线性预测器+标准化对称差",
    "xgb_shap": "XGBoost-SHAP", "xgb_pfi": "XGBoost-置换", "tabm_pfi": "TabM-置换",
    "deep_lasso": "Deep Lasso",
    "xgb_naive": "XGB-普通损失差",
    "xgb_midpoint": "XGB-中点对称差",
    "xgb_orbit": "XGB-随机对称差",
    "xgb_orbit_student": "XGB-标准化对称差",
    "tabm_naive": "TabM-普通损失差",
    "tabm_midpoint": "TabM-中点对称差",
    "tabm_orbit": "TabM-随机对称差",
    "tabm_orbit_student": "TabM-标准化对称差",
    "all_features": "全31项",
}
PAIRS = [
    ("tabm_orbit_student", "lasso_pair_q20"), ("tabm_orbit_student", "linear_orbit_student"),
    ("tabm_orbit_student", "tabm_midpoint"), ("tabm_orbit_student", "tabm_orbit"),
    ("tabm_orbit_student", "xgb_orbit_student"), ("tabm_orbit_student", "xgb_shap"),
    ("xgb_orbit_student", "lasso_pair_q20"), ("xgb_orbit_student", "linear_orbit_student"),
    ("xgb_orbit_student", "xgb_orbit"), ("xgb_orbit_student", "xgb_shap"),
]


def progress():
    out = {}
    for folder, required in (("real", 111), ("simulations", 800)):
        records = [read(p) for p in (RESULT / folder).glob("*.json")]
        out[folder] = {"count": len(records), "ok": sum(r["status"] == "ok" for r in records),
                       "required": required, "failed": [r for r in records if r["status"] != "ok"]}
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    save_json(RESULT / "coverage.json", out)
    return out


def audit_method_records(methods):
    assert set(methods) == set(METHODS)
    for record in methods.values():
        score = np.asarray(record["score"])
        assert score.shape == (31,) and np.isfinite(score).all()
        assert record["order"] == rank(score).tolist()
        for field, q in (("native", .2), ("native_q10", .1)):
            if record[field] is not None:
                assert record[field] == np.flatnonzero(score >= threshold(score, q)).tolist()


def audit_prediction(path, expected, frame, cached):
    data, meta = pd.read_csv(path), read(path.with_suffix(".json"))
    test = frame.loc[frame.split.eq("test")]
    for column in ("row_id", "seller_id", "target_month"):
        assert data[column].astype(str).tolist() == test[column].astype(str).tolist()
    assert np.allclose(data.log_gmv_next_month, test.log_gmv_next_month, rtol=1e-12, atol=1e-12)
    assert np.isfinite(data.prediction).all()
    assert meta["columns"] == sorted(expected)
    # Original predictions are float32; CSV reload and expm1 use float64.
    for key, value in metrics(test.log_gmv_next_month.to_numpy(), data.prediction.to_numpy()).items():
        assert np.isclose(value, meta["test"][key], rtol=2e-7, atol=1e-9), (path, key, value, meta["test"][key])
    cachekey = (tuple(meta["columns"]), meta["evaluator"])
    if cachekey in cached:
        assert np.array_equal(data.prediction, cached[cachekey])
    cached[cachekey] = data.prediction.to_numpy()


def audit():
    coverage = progress()
    assert all(v["count"] == v["required"] and v["ok"] == v["required"] for v in coverage.values())
    initial = read(RESULT / "initial_manifest.json")
    assert initial["protocol_sha"] == digest_file(ROOT / "protocol.json")
    assert initial["panel_sha"] == digest_file(V7 / "data_processed/seller_month_asof.parquet")
    assert initial["sampling_sha"] == digest_file(V7 / "data_processed/sampling_manifest.json")
    assert initial["predictor_parent_sha"] == digest_file(V7 / "code/models.py")
    frame = load_data()
    specs = read(V7 / "data_processed/sampling_manifest.json")
    for path in (RESULT / "real").glob("*.json"):
        record = read(path)
        assert record["protocol_sha"] == initial["protocol_sha"]
        assert not record["inference_labels_used_for_training"] and record["test_rows_read_by_selector"] == 0
        audit_method_records(record["methods"])
    for s in specs:
        r = read(RESULT / "real" / f"{s['run']}.json")
        d = frame.loc[frame.seller_id.isin(s["sellers"])]
        assert r["fit_row_hash"] == digest_array(d.loc[d.split.isin(["train", "tune"]), "row_id"].to_numpy())
        assert r["inference_row_hash"] == digest_array(d.loc[d.split.eq("rank"), "row_id"].to_numpy())
        assert not r["inference_labels_used_for_training"] and r["test_rows_read_by_selector"] == 0
        assert set(r["methods"]) == set(METHODS)
    for scene in P["simulation"]["scenarios"]:
        for rep in range(100):
            case = f"{scene['name']}_{rep:03d}"
            raw = dict(np.load(ROOT / "data" / f"{case}.npz"))
            r = read(RESULT / "simulations" / f"{case}.json")
            for field, key in (("x_hash", "x"), ("y_hash", "y"), ("xk_hash", "xk")):
                assert r[field] == digest_array(raw[key])
            assert r["protocol_sha"] == initial["protocol_sha"]
            audit_method_records(r["methods"])
            for name, m in r["methods"].items():
                assert sorted(m["order"]) == list(range(31))
                assert np.isfinite(m["rmse12"])
                for rule, found in (("top12", m["order"][:12]), ("top8", m["order"][:8])):
                    assert m[rule] == discovery_metrics(found, raw["truth"])
                if m["native"] is not None:
                    assert np.isfinite(m["rmse_native"])
                    assert m["native_metrics"] == discovery_metrics(m["native"], raw["truth"])
                    assert m["native10_metrics"] == discovery_metrics(m["native_q10"], raw["truth"])
                    assert np.array_equal(m["native"], np.flatnonzero(np.array(m["score"]) >= threshold(np.array(m["score"]))))
    ref = read(RESULT / "real/reference.json")["methods"]
    cached, predictions = {}, 0
    for method in METHODS + ["all_features"]:
        rules = ["all"] if method == "all_features" else ["top12", "top8"] + (
            ["native"] if ref[method]["native"] is not None else [])
        for ev in ("xgboost", "tabm"):
            for rule in rules:
                p = RESULT / "predictions" / f"{method}_{ev}_{rule}.csv"
                expected = (list(range(31)) if rule == "all" else ref[method]["native"] if rule == "native"
                            else ref[method]["order"][:int(rule[3:])])
                audit_prediction(p, expected, frame, cached)
                predictions += 1
    save_json(RESULT / "audit.json", {
        "passed": True, "case_counts": {"real": 111, "new_simulation_datasets": 800},
        "method_count": len(METHODS), "real_method_results": 111 * len(METHODS),
        "simulation_method_results": 800 * len(METHODS), "prediction_files": predictions,
        "all_metric_identities_checked": True, "all_methods_same_input_hash": True,
        "prediction_metrics_recomputed": True, "both_native_thresholds_recomputed": True,
        "prediction_metric_tolerance": {"rtol": 2e-7, "atol": 1e-9, "reason": "float32 predictions serialized to CSV"},
        "same_subsets_same_predictions": True, "new_simulation_seed_base": P["simulation"]["new_seed_base"],
        "code_hashes": {p.name: digest_file(p) for p in (ROOT / "code").glob("*.py")},
        "protocol_sha": initial["protocol_sha"], "model_training_independent_of_inference_responses": True,
        "real_population_fdr_not_established": True})


def holm(values):
    p = np.asarray(values)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def diff_stats(values):
    ci = cinterval(values)
    if ci["se"] == 0:
        pv = 1. if ci["mean"] == 0 else 0.
    else:
        pv = 2 * t.sf(abs(ci["mean"] / ci["se"]), len(values) - 1)
    return ci | {"p_two_sided": float(pv)}


def summarize_simulations():
    rows = []
    for path in sorted((RESULT / "simulations").glob("*.json")):
        r = read(path)
        for name, m in r["methods"].items():
            for rule in ("top12", "top8", "native", "native10"):
                if rule.startswith("native") and m["native"] is None:
                    continue
                vals = m[rule + "_metrics"] if rule.startswith("native") else m[rule]
                rows.append({"method": name, "scene": r["scene"]["name"], "rep": r["rep"], "rule": rule,
                             "theory_regime": r["scene"]["generator"] != "misspecified_gaussian_for_t",
                             "nominal_invalid_control": name.endswith("naive"), **vals,
                             "rmse": m["rmse12"] if rule == "top12" else m.get("rmse_native") if rule == "native" else None})
    data = pd.DataFrame(rows)
    data.to_csv(RESULT / "simulation_detail.csv", index=False)
    summaries = []
    for (scene, method, rule), part in data.groupby(["scene", "method", "rule"]):
        row = {"scene": scene, "method": method, "rule": rule}
        for key in ("FDP", "power", "size", "rmse"):
            vals = part[key].dropna()
            row.update({f"{key}_{k}": v for k, v in cinterval(vals).items()} if len(vals) else
                       {f"{key}_{k}": None for k in ("mean", "sd", "se", "ci_low", "ci_high", "n")})
        row["nonempty_fraction"] = float((part["size"] > 0).mean())
        if scene == "global_null" and rule.startswith("native"):
            successes, n = (part["size"] > 0).sum(), len(part)
            row["null_fdr_cp_low"] = 0 if successes == 0 else beta.ppf(.025, successes, n - successes + 1)
            row["null_fdr_cp_high"] = 1 if successes == n else beta.ppf(.975, successes + 1, n - successes)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULT / "simulation_summary.csv", index=False)
    paired = []
    for (scene, rule), part in data.groupby(["scene", "rule"]):
        for metric in ("FDP", "power", "rmse"):
            matrix = part.pivot(index="rep", columns="method", values=metric)
            group = []
            for a, b in PAIRS:
                if a not in matrix or b not in matrix:
                    continue
                diff = (matrix[a] - matrix[b]).dropna()
                if len(diff) != 100:
                    continue
                group.append({"scene": scene, "rule": rule, "metric": metric,
                              "candidate": a, "comparator": b, **diff_stats(diff)})
            if group:
                adjusted = holm([r["p_two_sided"] for r in group])
                for entry, pv in zip(group, adjusted):
                    entry["holm_p_within_scene_rule_metric"] = pv
                    paired.append(entry)
    pd.DataFrame(paired).to_csv(RESULT / "simulation_paired.csv", index=False)


def stability(mvr=False):
    real_folder = RESULT / ("real_mvr" if mvr else "real")
    prefix = "mvr_" if mvr else ""
    specs = read(V7 / "data_processed/sampling_manifest.json")
    ref = read(real_folder / "reference.json")["methods"]
    rows, native_rows, seedrows = [], [], []
    for s in specs:
        r = read(real_folder / f"{s['run']}.json")["methods"]
        for name in METHODS:
            rows.append({"method": name, "run": s["run"], "group": s["group"], "fraction": s["fraction"],
                         "half": s["half"], "J": jaccard(ref[name]["order"][:12], r[name]["order"][:12]),
                         "set": json.dumps(r[name]["order"][:12])})
            if r[name]["native"] is not None:
                native_rows.append({"method": name, "run": s["run"], "fraction": s["fraction"],
                                    "size": len(r[name]["native"]), "empty": not r[name]["native"]})
    data = pd.DataFrame(rows)
    summary, complementary, paired = [], [], []
    for (method, frac), part in data.groupby(["method", "fraction"]):
        summary.append({"method": method, "fraction": frac,
                        **cinterval(part.groupby("group").J.mean()),
                        "nogueira": nogueira([json.loads(v) for v in part["set"]])})
        if frac == .5:
            for group, sub in part.groupby("group"):
                a, b = sub["set"].map(json.loads)
                complementary.append({"method": method, "group": group, "J": jaccard(a, b)})
    cp = pd.DataFrame(complementary)
    references = data.loc[data.fraction.eq(.5)].groupby(["method", "group"]).J.mean().unstack(0)
    complements = cp.pivot(index="group", columns="method", values="J")
    for metric, matrix in (("reference", references), ("complementary", complements)):
        group = []
        for a, b in PAIRS:
            group.append({"candidate": a, "comparator": b, "metric": metric, **diff_stats(matrix[a] - matrix[b])})
        for row, pv in zip(group, holm([r["p_two_sided"] for r in group])):
            row["holm_p"] = pv
            paired.append(row)
    for i in range(10):
        r = read(real_folder / f"seedonly_{i:02d}.json")["methods"]
        for name in METHODS:
            seedrows.append({"method": name, "seed_batch": i, "J": jaccard(ref[name]["order"][:12], r[name]["order"][:12])})
    data.to_csv(RESULT / f"{prefix}stability_detail.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULT / f"{prefix}stability_summary.csv", index=False)
    cp.to_csv(RESULT / f"{prefix}stability_complementary.csv", index=False)
    pd.DataFrame(paired).to_csv(RESULT / f"{prefix}stability_paired.csv", index=False)
    pd.DataFrame(seedrows).to_csv(RESULT / f"{prefix}algorithm_randomness.csv", index=False)
    pd.DataFrame(native_rows).to_csv(RESULT / f"{prefix}native_sizes.csv", index=False)


def predictions(mvr=False):
    folder = RESULT / ("predictions_mvr" if mvr else "predictions")
    prefix = "mvr_" if mvr else ""
    extra_pairs = ([(f"{model}_{mode}", "xgb_shap")
                    for model in ("xgb", "tabm") for mode in ("naive", "midpoint", "orbit")] if mvr else [])
    comparisons = PAIRS + extra_pairs
    summary, paired = [], []
    for p in folder.glob("*.json"):
        r = read(p)
        summary.append({"method": r["method"], "evaluator": r["evaluator"], "rule": r["rule"],
                        "size": len(r["columns"]), **r["test"]})
    pd.DataFrame(summary).to_csv(RESULT / f"{prefix}prediction_summary.csv", index=False)
    for rule in (("top12",) if mvr else ("top12", "top8")):
        for ev in ("xgboost", "tabm"):
            df = pd.read_csv(folder / f"xgb_shap_{ev}_{rule}.csv")
            ids, idx = np.unique(df.seller_id, return_inverse=True)
            sizes = np.bincount(idx)
            rng = np.random.default_rng(916)
            draws = rng.integers(0, len(ids), (5000, len(ids)))
            boot, point = {}, {}
            for name in METHODS:
                r = pd.read_csv(folder / f"{name}_{ev}_{rule}.csv")
                assert np.array_equal(r.row_id, df.row_id)
                error = (r.prediction.to_numpy() - df.log_gmv_next_month.to_numpy())**2
                sums = np.bincount(idx, weights=error)
                boot[name] = np.sqrt(sums[draws].sum(1) / sizes[draws].sum(1))
                point[name] = np.sqrt(error.mean())
            for a, b in comparisons:
                diff = boot[a] - boot[b]
                paired.append({"candidate": a, "comparator": b, "evaluator": ev, "rule": rule,
                    "difference": point[a] - point[b], "ci_low": np.quantile(diff, .025),
                    "ci_high": np.quantile(diff, .975), "bonf_low": np.quantile(diff, .025 / (2 * len(comparisons))),
                    "bonf_high": np.quantile(diff, 1 - .025 / (2 * len(comparisons))),
                    "comparison_status": "posthoc_generator_diagnostic" if (a, b) in extra_pairs else "main_comparison",
                    "interpretation": "conditional on previously observed test months and fixed models"})
    pd.DataFrame(paired).to_csv(RESULT / f"{prefix}prediction_paired.csv", index=False)


def generator_diagnostic(mvr=False):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score
    filename = "real_generator_pairs_mvr.npz" if mvr else "real_generator_pairs.npz"
    arrays = dict(np.load(ROOT / "data" / filename))
    raw = load_data()
    frame = raw.set_index("row_id").loc[arrays["row_id"]]
    rng = np.random.default_rng(1701)
    unique = rng.permutation(frame.seller_id.unique())
    train = frame.seller_id.isin(unique[:int(len(unique) * .7)]).to_numpy()
    x, xk = arrays["x"], arrays["xk"]
    base = np.column_stack([x, xk])
    rows = []
    for fraction in (.25, .5, 1.0):
        chosen = rng.choice(31, int(31 * fraction), replace=False)
        swapped = base.copy()
        swapped[:, chosen], swapped[:, chosen + 31] = base[:, chosen + 31], base[:, chosen]
        clf = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=8, n_jobs=1, random_state=11)
        clf.fit(np.concatenate([base[train], swapped[train]]),
                np.r_[np.zeros(train.sum()), np.ones(train.sum())])
        test = np.concatenate([base[~train], swapped[~train]])
        y = np.r_[np.zeros((~train).sum()), np.ones((~train).sum())]
        auc = roc_auc_score(y, clf.predict_proba(test)[:, 1])
        rows.append({"swap_fraction": fraction, "seller_disjoint_auc": auc,
                     "absolute_auc_departure_from_half": abs(auc - .5)})
    save_json(RESULT / ("mvr_generator_diagnostic.json" if mvr else "real_generator_diagnostic.json"), {
        "tests": rows, "interpretation": "engineering diagnostic, not a calibrated hypothesis test",
        "passing_not_proof": True, "theoretical_real_fdr_guarantee": False,
        "train_test_split_by_seller": True})


def mvr_summary():
    paths = list((RESULT / "real_mvr").glob("*.json"))
    assert len(paths) == 111
    for path in paths:
        r = read(path)
        original = read(RESULT / "real" / path.name)
        assert r["status"] == "ok"
        audit_method_records(r["methods"])
        assert r["protocol_sha"] == digest_file(ROOT / "protocol.json")
        assert not r["inference_labels_used_for_training"] and r["test_rows_read_by_selector"] == 0
        assert r["fit_row_hash"] == original["fit_row_hash"]
        assert r["inference_row_hash"] == original["inference_row_hash"]
        assert r["generator"]["extension_protocol_sha"] == digest_file(ROOT / "mvr_extension_protocol.json")
        for name in ("xgb_shap", "xgb_pfi", "tabm_pfi", "deep_lasso"):
            assert r["methods"][name]["order"] == original["methods"][name]["order"]
    ref = read(RESULT / "real_mvr/reference.json")["methods"]
    frame, cached = load_data(), {}
    for path in (RESULT / "predictions").glob("*.csv"):
        meta = read(path.with_suffix(".json"))
        audit_prediction(path, meta["columns"], frame, cached)
    for method in METHODS + ["all_features"]:
        for ev in ("xgboost", "tabm"):
            rule = "all" if method == "all_features" else "top12"
            expected = list(range(31)) if rule == "all" else ref[method]["order"][:12]
            file = RESULT / "predictions_mvr" / f"{method}_{ev}_{rule}.csv"
            audit_prediction(file, expected, frame, cached)
    stability(mvr=True)
    predictions(mvr=True)
    generator_diagnostic(mvr=True)
    save_json(RESULT / "mvr_audit.json", {"passed": True, "real_cases": 111,
        "method_results": 111 * len(METHODS), "top12_prediction_files": 2 * len(METHODS),
        "all_feature_prediction_files": 2, "total_prediction_files": 30,
        "prediction_metrics_recomputed": True, "same_subsets_same_predictions_across_generators": True,
        "all_original_seller_samples_preserved": True, "non_knockoff_rankings_unchanged": True,
        "posthoc_extension": True, "no_new_simulations_for_extension": True,
        "extension_protocol_sha": digest_file(ROOT / "mvr_extension_protocol.json"),
        "code_hashes": {p.name: digest_file(p) for p in (ROOT / "code").glob("*.py")}})


def markdown(headers, rows):
    def line(items):
        return "| " + " | ".join(str(x).replace("|", "/") for x in items) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *[line(r) for r in rows]])


def report():
    pred = pd.read_csv(RESULT / "prediction_summary.csv")
    sim = pd.read_csv(RESULT / "simulation_summary.csv")
    stab = pd.read_csv(RESULT / "stability_summary.csv")
    ref = read(RESULT / "real/reference.json")["methods"]
    rows = []
    exact_nonnull = [s["name"] for s in P["simulation"]["scenarios"]
                     if s["signals"] and s["generator"] != "misspecified_gaussian_for_t"]
    for method in METHODS:
        r = pred.loc[pred.method.eq(method) & pred.rule.eq("top12")].set_index("evaluator")
        s = stab.loc[stab.method.eq(method) & stab.fraction.eq(.5)].iloc[0]
        m = sim.loc[sim.method.eq(method) & sim.rule.eq("top12") & sim.scene.isin(exact_nonnull)]
        rows.append([LABELS[method], f"{r.loc['xgboost','rmse_log']:.4f}",
                     f"{r.loc['tabm','rmse_log']:.4f}", f"{s['mean']:.4f}", f"{s.nogueira:.4f}",
                     f"{m.FDP_mean.mean():.4f}", f"{m.power_mean.mean():.4f}",
                     len(ref[method]["native"]) if ref[method]["native"] is not None else "无原生规则"])
    overview = markdown(["方法", "XGB-RMSE", "TabM-RMSE", "半样本J", "Nogueira", "FDP@12", "Power@12", "真实原生数"], rows)
    pd.DataFrame(rows, columns=["method", "XGB_RMSE", "TabM_RMSE", "J", "Nogueira", "FDP12", "Power12", "native_size"]).to_csv(
        RESULT / "comparison_overview.csv", index=False)
    scene_text = []
    for scene in P["simulation"]["scenarios"]:
        rows = []
        for method in METHODS:
            records = sim.loc[sim.method.eq(method) & sim.scene.eq(scene["name"]) & sim.rule.eq("native")]
            if records.empty:
                continue
            r = records.iloc[0]
            rows.append([LABELS[method], f"{r.FDP_mean:.4f}", f"{r.FDP_se:.4f}",
                         "不定义" if pd.isna(r.power_mean) else f"{r.power_mean:.4f}",
                         f"{r['size_mean']:.2f}", f"{r.nonempty_fraction:.2f}",
                         f"{r.rmse_mean:.4f}"])
        scene_text += [f"### {scene['name']}", markdown(
            ["方法", "FDP", "MCSE", "Power", "发现数", "非空比例", "名单RMSE"], rows)]
    scenes = "\n\n".join(scene_text)
    diag = read(RESULT / "real_generator_diagnostic.json")
    diagnostics = markdown(["交换比例", "卖家隔离AUC"], [
        [r["swap_fraction"], f"{r['seller_disjoint_auc']:.4f}"] for r in diag["tests"]])
    primary = "tabm_orbit_student"
    paired = pd.read_csv(RESULT / "simulation_paired.csv")
    findings = []
    for scene in exact_nonnull:
        r = sim.loc[sim.method.eq(primary) & sim.rule.eq("native") & sim.scene.eq(scene)].iloc[0]
        base = sim.loc[sim.method.eq("linear_orbit_student") & sim.rule.eq("native") & sim.scene.eq(scene)].iloc[0]
        dif = paired.loc[paired.candidate.eq(primary) & paired.comparator.eq("linear_orbit_student") &
                         paired.rule.eq("native") & paired.metric.eq("power") & paired.scene.eq(scene)].iloc[0]
        findings.append([scene, f"{r.FDP_mean:.4f}", f"{r.power_mean:.4f}",
                         f"{base.power_mean:.4f}", f"{dif['mean']:+.4f}",
                         f"{dif.holm_p_within_scene_rule_metric:.4g}"])
    findings_table = markdown(["场景", "TabM-FDP", "TabM-Power", "线性-Power", "Power差", "Holm p"], findings)
    indexed = sim.set_index(["scene", "method", "rule"])
    strong = indexed.loc[("nonlinear_strong", primary, "native")]
    strong_linear = indexed.loc[("nonlinear_strong", "linear_orbit_student", "native")]
    strong_orbit = indexed.loc[("nonlinear_strong", "tabm_orbit", "native")]
    interaction = indexed.loc[("pure_interactions", primary, "native")]
    interaction_linear = indexed.loc[("pure_interactions", "linear_orbit_student", "native")]
    correlated = indexed.loc[("high_correlation", primary, "native")]
    correlated_linear = indexed.loc[("high_correlation", "linear_orbit_student", "native")]
    null = indexed.loc[("global_null", primary, "native")]
    main_pred = pred.loc[pred.rule.eq("top12") & pred.evaluator.eq("xgboost")].set_index("method")
    main_stab = stab.loc[stab.fraction.eq(.5)].set_index("method")
    qrows = []
    for scene in P["simulation"]["scenarios"]:
        a = indexed.loc[(scene["name"], primary, "native")]
        b = indexed.loc[(scene["name"], primary, "native10")]
        qrows.append([scene["name"], f"{a.FDP_mean:.4f}",
                      "不定义" if pd.isna(a.power_mean) else f"{a.power_mean:.4f}",
                      f"{b.FDP_mean:.4f}", "不定义" if pd.isna(b.power_mean) else f"{b.power_mean:.4f}",
                      f"{b['size_mean']:.2f}"])
    qtable = markdown(["场景", "q=.2 FDP", "q=.2 Power", "q=.1 FDP", "q=.1 Power", "q=.1发现数"], qrows)
    timing = []
    for folder in ("real", "simulations", "real_mvr"):
        seconds = [read(p)["seconds"] for p in (RESULT / folder).glob("*.json")]
        if seconds:
            timing.append({"scope": folder, "completed_cases": len(seconds),
                           "median_case_seconds": float(np.median(seconds)),
                           "sum_case_elapsed_hours": float(np.sum(seconds) / 3600)})
    pd.DataFrame(timing).to_csv(RESULT / "runtime_summary.csv", index=False)
    time_table = markdown(["运行范围", "完成案例", "每案例秒数中位数", "案例耗时合计/小时"],
                          [[r["scope"], r["completed_cases"], f'{r["median_case_seconds"]:.1f}',
                            f'{r["sum_case_elapsed_hours"]:.2f}'] for r in timing])
    extension = ""
    if (RESULT / "mvr_audit.json").exists():
        mp = pd.read_csv(RESULT / "mvr_prediction_summary.csv")
        ms = pd.read_csv(RESULT / "mvr_stability_summary.csv")
        mn = pd.read_csv(RESULT / "mvr_native_sizes.csv")
        mr = read(RESULT / "real_mvr/reference.json")["methods"]
        rows = []
        for method in METHODS:
            pp = mp.loc[mp.method.eq(method) & mp.rule.eq("top12")].set_index("evaluator")
            ss = ms.loc[ms.method.eq(method) & ms.fraction.eq(.5)].iloc[0]
            native = mr[method]["native"]
            npart = mn.loc[mn.method.eq(method) & mn.fraction.eq(.5)]
            rows.append([LABELS[method], f"{pp.loc['xgboost','rmse_log']:.4f}",
                         f"{pp.loc['tabm','rmse_log']:.4f}", f"{ss['mean']:.4f}", f"{ss.nogueira:.4f}",
                         len(native) if native is not None else "无",
                         f"{npart['empty'].mean():.3f}" if len(npart) else "不适用"])
        et = markdown(["方法", "XGB-RMSE", "TabM-RMSE", "半样本J", "Nogueira", "名义发现数", "半样本空集率"], rows)
        dg = read(RESULT / "mvr_generator_diagnostic.json")["tests"]
        dt = markdown(["交换比例", "卖家隔离AUC"], [
            [r["swap_fraction"], f"{r['seller_disjoint_auc']:.4f}"] for r in dg])
        mprimary = mp.loc[mp.method.eq(primary) & mp.evaluator.eq("xgboost")].iloc[0]
        mprimary_stab = ms.loc[ms.method.eq(primary) & ms.fraction.eq(.5)].iloc[0]
        mmiddle = mp.loc[mp.method.eq("tabm_midpoint") & mp.evaluator.eq("xgboost")].iloc[0]
        mmiddle_stab = ms.loc[ms.method.eq("tabm_midpoint") & ms.fraction.eq(.5)].iloc[0]
        mpaired = pd.read_csv(RESULT / "mvr_prediction_paired.csv")
        middle_difference = mpaired.loc[mpaired.candidate.eq("tabm_midpoint") &
                                        mpaired.comparator.eq("xgb_shap") &
                                        mpaired.evaluator.eq("xgboost")].iloc[0]
        extension = f"""
## 9. 事后追加：MVR生成器敏感性

主协议的真实等相关S约0.00334，约49.4%的真假数值完全相同，比较信息很弱。一次参考样本检查已看到MVR的名义发现数及测试RMSE变化，随后才制定mvr_extension_protocol.json，追加相同111个真实案例及两种评价器的Top-12分析，不改动前述主表。追加动机受到已见结果启发；不能把“未搜索生成器参数或修改q”误写成“生成器选择完全未受测试表现影响”。

{et}

{dt}

这是观察主实验结果后追加的诊断，不是预先注册确认。MVR为既有方法，不列为本轮原创；相同样本、基础模型、阈值保持不变，四项不使用Knockoff的基线排名逐案例与主实验一致。只重新检验真实样本，不把前述精确等相关模拟结果误写成MVR实测FDR。发现数增加不证明原生FDR正确；估计Copula及跨期卖家依赖的限制依然存在。

主候选在MVR下的XGB评价RMSE为{mprimary.rmse_log:.4f}，半样本J为{mprimary_stab['mean']:.4f}；TabM中点版本分别为{mmiddle.rmse_log:.4f}和{mmiddle_stab['mean']:.4f}。中点版本只属于已登记的机制消融，不能事后改称预定主候选。主实验、追加实验不能各抽一个最好的指标拼成“总体最优方法”。

为避免把小幅数值差当成确定收益，另对追加实验全部六个普通/中点/随机背景版本补算相对SHAP的配对区间，标记为事后诊断，不改变主比较族。中点TabM相对SHAP的XGB评价RMSE差为{middle_difference.difference:+.4f}，卖家簇Bootstrap点态95%区间为[{middle_difference.ci_low:+.4f}, {middle_difference.ci_high:+.4f}]。这不是新测试集上的确认性检验；区间包含0时不能声称已证实优于SHAP，也不能声称两者已证明等价。
"""
    text = f"""# 交换对称成对损失：方法原型与完整实测

## 1. 研究定位

本轮没有将模型改名当作原创，而是构造了一个固定预测器上的成对损失统计量，要求任意真假列交换时，被交换统计量反号、其他统计量不变。它是候选方法原型，不是已发表新算法，也尚未完成原创性系统检索。

研究已完成111个Olist开发样本实验、800份全新种子的模拟数据，每份都计算14种方法或消融，共1554个真实方法结果和11200个模拟方法结果。网络使用三个训练种子；未用旧模拟结果替代新计算。模型权重、日志、协议、数据摘要及失败检查可复现。

与此前不同，基础预测器不能用评分期响应早停。训练/早停完成后固定预测器，再在独立评分期产生排名。因此即使方法名称相同，本表也不能直接与旧实验未重新计算的RMSE混排。

## 2. 算法具体变化

1. 在每个真假数值对内取min/max，按固定随机掩码生成八组不区分真假的背景。
2. 对特征j分别放入真实值和Knockoff值，计算平方预测损失差；其他变量始终使用相同的对称背景。
3. 对背景、卖家内观测及卖家平均，形成W；标准化版再除以卖家均值标准误加常数。
4. 使用既有Knockoff+阈值产生原生名单，同时单独报告Top-12/Top-8预算名单。

本轮拆分普通替换、中点背景、随机对称背景、标准化四种评分，并分别接入同一批已训练XGBoost和官方TabM。另有SHAP、普通置换、Deep Lasso、评分集成对Lasso，以及使用同样训练样本的线性预测器。标准化不是t检验。

完整代数证明、先行文献以及独立样本/精确生成器前提见后附方法说明。符号翻转验证误差为零；这不证明Olist生成器精确。

## 3. 公平设计与解释边界

所有真实方法共享同一卖家清单：30组互补50%样本、20次70%、20次80%，另10组种子变化及完整参考。固定预算K=12，敏感性K=8。最终预测统一特征列顺序、模型参数、种子和时间窗口，相同变量集合复用相同预测。

真实面板共13,754行、2,723个卖家、31项指标；train/tune/rank/test分别为6,567/1,831/1,943/3,413行。响应为下一自然月签收确认商品金额的log1p，不含运费，不是按购买月回溯的最终送达GMV。RMSE、MAE和R²在log1p尺度计算，WAPE在金额尺度计算。静态卖家/商品信息的历史可见性仍有假设，不能将其包装为已经部署的实时预测系统。

新模拟800份分为八场景各100次，n=2000、p=31；训练1200行、评分400行、最终预测测试400行。七场景使用已知分布下的精确Knockoff，t错设场景故意使用不正确的高斯生成器。各场景和失败结果不按有利程度筛除。

成对Lasso只在评分样本拟合，其400行标签预算与基础模型在前1200行拟合的方式不同。所以主要还要看同样使用1200行训练、相同评分器的线性预测器对照。原生q=0.20，q=0.10敏感性在同一个W上重算。

真实数据存在估计Copula失配及同卖家跨期依赖，原生名单只称“名义阈值发现”，不得说实际FDR已被证明受控。2018年测试期曾多次查看，结果属探索性，不是独立外部确认。普通替换损失差不满足本轮要求的完整符号翻转，对它应用名义Knockoff阈值只是压力对照，不是CPI文献正式检验的复现。

## 4. 同预算总览

{overview}

FDP@12和Power@12仅为六个精确生成器非零信号场景的等权平均；全无效和生成器错设另列。固定K时二者相关，不当作两份独立成功证据。真实原生发现为0也保留，预测采用常数均值作为空集基线。

## 5. 预定主候选与线性对照

{findings_table}

**已得到的判断。** 主候选在强非线性场景的Power为{strong.power_mean:.3f}，匹配线性对照为{strong_linear.power_mean:.3f}，同时平均FDP为{strong.FDP_mean:.3f}；纯交互场景的Power为{interaction.power_mean:.3f}对{interaction_linear.power_mean:.3f}。这支持利用非线性预测器发现线性评分遗漏的变量，但不证明优于所有已有非线性Knockoff方法。

**标准化改动没有获得支持。** 同一TabM预测器的未标准化随机对称差在强非线性场景Power为{strong_orbit.power_mean:.3f}，高于预定标准化版本。标准化版本在高相关线性场景为{correlated.power_mean:.3f}，低于匹配线性对照的{correlated_linear.power_mean:.3f}。多个差异通过同场景Holm校正，不能把标准化描述成普遍提升检出率的创新。

**真实应用未形成优势。** 主实验Top-12、统一XGB评价下，主候选RMSE为{main_pred.loc[primary, 'rmse_log']:.4f}，SHAP为{main_pred.loc['xgb_shap', 'rmse_log']:.4f}；半样本J分别为{main_stab.loc[primary, 'mean']:.4f}和{main_stab.loc['xgb_shap', 'mean']:.4f}。主候选在这两个指标上均更差，不能以模拟Power的进步替代真实预测和稳定性的失败。

Holm校正在同场景、同指标、同规则的预定比较族内进行；不覆盖跨所有场景择优的结论。FDR是平均FDP的总体期望，单次FDP不必小于q。100次模拟只提供带Monte Carlo误差的估计，不是理论证明。

## 6. 全部原生发现结果

{scenes}

全无效场景Power没有定义；其FDR等于选出非空集合的概率。准确二项区间在simulation_summary.csv中报告，不把100次全未发现说成真实错误率必为0。发现数不同的原生名单与强制12项名单不混比。

主候选全无效场景为100次中{int(round(null.FDP_mean * 100))}次非空，估计FDR={null.FDP_mean:.3f}，准确95%区间为[{null.null_fdr_cp_low:.3f}, {null.null_fdr_cp_high:.3f}]。线性稀疏场景的TabM中点平均FDP略超过0.20，仍须连同Monte Carlo标准误如实保留，不能宣称每个有限模拟均值都严格低于目标。

### 同一主候选的阈值敏感性

{qtable}

这些是相同W的不同阈值，不是重新训练的模型。q=.1至少需要10个正发现才可能非空，稀疏信号场景可能因此损失Power。t错设行只有经验意义，即使平均FDP低于目标也不能恢复不存在的理论保证。

## 7. 真实生成器诊断

{diagnostics}

分类训练与测试按卖家分离，原样本及交换样本成对划入同一侧。该AUC仅为工程诊断，不提供检验p值；接近0.5也不是交换性证明。因为评分所用卖家可能出现在模型训练历史中，独立样本理论不能直接外推。

## 8. 怎样评价论文可行性

有明确研究问题、算法定义、代数性质、机制消融和完整数据实测，比不断拼接网络更具可讨论性。但是否足以成为学位论文方法贡献，需要分别判断：原创性是否与先行文献区分、目标场景是否真实改善、真实数据是否有可使用结果。不能用大批训练次数替代任何一项。

本轮的明确判断是：可以作为“模型无关损失评分的对称化、非线性检出及生成器敏感性”方法研究原型，但尚不能作为“已证明原创且在电商数据上全面改进”的定稿成果。预定标准化版本不宜成为成功创新的核心叙事；简单对称化是否与CPI、Semi-knockoffs等文献实质不同，还需逐公式对照，并补足近期非线性受控筛选基线和第二个未使用过的真实数据集。

可以讨论的暂定题目是《面向电商经营指标筛选的交换对称损失统计量及其稳健性研究》。这个题目描述研究问题，不承诺方法优于一切基线。当前不能保证其创新程度满足学位要求，更不能保证答辩通过。

### 计算成本与复现审计

{time_table}

每案例同时训练和计算多种共享预测器/统计量。案例耗时合计是各进程计时之和，不是整轮墙钟耗时或CPU核心小时，不能拆算为某方法独有成本。本轮未保存逐评分方法的独立计时，因此不提供虚构的速度排名。

审计对预测RMSE、MAE、R²、WAPE从逐行预测重算，并检查样本哈希、两种原生阈值、有限分数和相同集合预测一致性。50%稳定性区间按30组互补样本计算，不把60个半样本当独立重复。完整补充指标、配对区间及代码摘要保存在results目录；生成报告没有重调模型参数或改动冻结协议。

逐行预测最初含float32输出，CSV重读后按float64复算金额指数函数会产生约2e-8的差异；核对采用rtol=2e-7、atol=1e-9，不影响报告四位小数。NumPy/BLAS矩阵乘法警告保留在运行记录中，不通过关闭警告隐藏。附加的10万行精确高斯检查同时核验随机背景协方差，并用不调用BLAS的显式einsum交叉核对采样矩阵乘法；它不是真实Olist生成器合格证明。

{extension}
"""
    (ROOT / "研究报告.md").write_text(text)
    print(overview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["progress", "diagnostic", "audit", "summary", "report", "mvr-summary"])
    args = parser.parse_args()
    if args.stage == "progress":
        progress()
    elif args.stage == "diagnostic":
        generator_diagnostic()
    elif args.stage == "audit":
        audit()
    elif args.stage == "summary":
        audit()
        summarize_simulations()
        stability()
        predictions()
        generator_diagnostic()
        report()
    elif args.stage == "mvr-summary":
        mvr_summary()
        report()
    else:
        report()
