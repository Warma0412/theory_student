"""Audit common inputs, summarize paired experiments, and freeze development choice."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import (ROOT, FEATURES, KS, SEEDS, Preprocessor, load_data, split_arrays,
                    digest_array, digest_file, save_json, jaccard, threshold, ebh)
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.covariance import LedoitWolf
from knockpy.knockoffs import GaussianSampler
from models import xgb_model, lasso_competition

COMPARATORS = ["copula_mvr", "elastic_net", "stability_selection", "shadow_trees",
               "xgboost_shap", "deep_lasso", "tabm", "vtfs"]
LABELS = {
    "copula_mvr": "Copula-MVR",
    "elastic_net": "Elastic Net",
    "stability_selection": "稳定性选择",
    "shadow_trees": "影子变量树",
    "xgboost_shap": "XGBoost-SHAP",
    "deep_lasso": "Deep Lasso",
    "tabm": "TabM-置换",
    "vtfs": "VTFS-定额适配",
    "all_features": "全维基线",
}


def read(path):
    return json.loads(Path(path).read_text())


def freeze_choice():
    dest = ROOT / "results/development_choice.json"
    if dest.exists():
        return
    df = load_data()
    assert not (ROOT / "results/predictions").exists(), "Test predictions must not precede this freeze"
    x, y, xv, yv, _, _ = split_arrays(df, ("train", "tune"), "rank")
    pre = Preprocessor().fit(x)
    x, xv = pre.transform(x), pre.transform(xv)
    params = read(ROOT / "results/configs/xgboost_shap.json")["params"]
    rows = []
    for method in COMPARATORS:
        ref = read(ROOT / "results/runs" / method / "reference.json")
        for k in KS:
            idx = ref["order"][:k]
            pred = np.mean([xgb_model(params, seed).fit(x[:, idx], y).predict(xv[:, idx])
                            for seed in SEEDS], axis=0)
            rows.append({"method": method, "k": k, "validation_rmse": float(np.mean((pred - yv)**2)**.5)})
    table = pd.DataFrame(rows)
    best12 = table.loc[table.k.eq(12)].sort_values("validation_rmse").iloc[0].to_dict()
    best = table.sort_values("validation_rmse").iloc[0]
    acceptable = table.loc[table.validation_rmse <= best.validation_rmse * 1.01]
    compact = acceptable.sort_values(["k", "validation_rmse"]).iloc[0].to_dict()
    table.to_csv(ROOT / "results/development_comparison.csv", index=False)
    save_json(dest, {"primary_k": 12, "primary_method": best12["method"], "selected_at_primary_k": best12,
                     "compact_rule": "smallest K within 1% of best development RMSE",
                     "compact_choice": compact, "test_read": False,
                     "protocol_sha256": digest_file(ROOT / "protocol.json"),
                     "interpretation": "exploratory development selection; external independent confirmation is still needed"})
    print(json.dumps(read(dest), ensure_ascii=False, indent=2))


def cinterval(values):
    a = np.array(values, dtype=float)
    mean = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.
    width = t.ppf(.975, len(a) - 1) * se if len(a) > 1 else 0.
    return {"mean": float(mean), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.,
            "se": float(se), "ci_low": float(mean - width), "ci_high": float(mean + width), "n": len(a)}


def nogueira(sets, p=31):
    z = np.zeros((len(sets), p))
    for i, subset in enumerate(sets):
        z[i, list(subset)] = 1
    prob = z.mean()
    denom = prob * (1 - prob)
    return float(1 - z.var(axis=0, ddof=1).mean() / denom) if denom else 1.


def summarize_stability(require_complete=True):
    specs = read(ROOT / "data_processed/sampling_manifest.json")
    detail, summary, coverage, pairrows, algorithm = [], [], [], [], []
    df = load_data()
    for method in COMPARATORS:
        directory = ROOT / "results/runs" / method
        ref = read(directory / "reference.json")
        count = 0
        for spec in specs:
            path = directory / (spec["run"] + ".json")
            if not path.exists():
                continue
            row = read(path)
            if row["status"] != "ok":
                continue
            count += 1
            subdf = df.loc[df.seller_id.isin(spec["sellers"])]
            assert row["fit_row_hash"] == digest_array(
                subdf.loc[subdf.split.isin(["train", "tune"]), "row_id"].to_numpy())
            for k in KS:
                current, base = row["order"][:k], ref["order"][:k]
                detail.append({"method": method, "run": spec["run"], "group": spec["group"],
                               "fraction": spec["fraction"], "half": spec["half"], "k": k,
                               "reference_jaccard": jaccard(current, base),
                               "features": json.dumps(current),
                               "native_count": len(row["native"]) if row["native"] is not None else None,
                               "seconds": row["total_seconds"]})
        coverage.append({"method": method, "completed": count, "required": 100})
        for batch in range(10):
            path = directory / f"seedonly_{batch:02d}.json"
            if path.exists() and read(path)["status"] == "ok":
                row = read(path)
                for k in KS:
                    algorithm.append({"method": method, "batch": batch, "k": k,
                                      "reference_jaccard": jaccard(ref["order"][:k], row["order"][:k])})
    details = pd.DataFrame(detail)
    pd.DataFrame(coverage).to_csv(ROOT / "results/stability_coverage.csv", index=False)
    if require_complete:
        assert all(r["completed"] == 100 for r in coverage), coverage
        assert len(algorithm) == len(COMPARATORS) * 10 * len(KS)
    for (method, fraction, k), part in details.groupby(["method", "fraction", "k"]):
        grouped = part.groupby("group").reference_jaccard.mean()
        entry = dict(method=method, fraction=fraction, k=k, metric="reference_jaccard")
        entry.update(cinterval(grouped.to_numpy()))
        entry["nogueira"] = nogueira([json.loads(v) for v in part.features])
        summary.append(entry)
        if fraction == .5:
            for group, ab in part.groupby("group"):
                if len(ab) == 2:
                    aa, bb = [json.loads(v) for v in ab.sort_values("half").features]
                    pairrows.append({"method": method, "group": group, "k": k,
                                     "complementary_jaccard": jaccard(aa, bb)})
    details.to_csv(ROOT / "results/stability_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "results/stability_summary.csv", index=False)
    pd.DataFrame(pairrows).to_csv(ROOT / "results/complementary_pair_runs.csv", index=False)
    pd.DataFrame(algorithm).to_csv(ROOT / "results/algorithm_randomness.csv", index=False)
    for (k,), pairpart in pd.DataFrame(pairrows).groupby(["k"]):
        pivot = pairpart.pivot(index="group", columns="method", values="complementary_jaccard")
        for left in pivot:
            for right in pivot:
                if left < right:
                    diff = (pivot[left] - pivot[right]).dropna()
                    pairrows.append({"comparison": f"{left} - {right}", "k": k, **cinterval(diff)})
    pd.DataFrame([r for r in pairrows if "comparison" in r]).to_csv(
        ROOT / "results/paired_stability_differences.csv", index=False)


def summarize_simulations(require_complete=True):
    rows, counts = [], []
    for method in COMPARATORS:
        directory = ROOT / "results/simulation_runs" / method
        ok = 0
        for path in sorted(directory.glob("*.json")):
            row = read(path)
            if row["status"] != "ok":
                continue
            ok += 1
            for metric in row["metrics"]:
                rows.append({"method": method, "scenario": row["scenario"]["name"],
                             "rep": row["rep"], "rmse_at_12": row["rmse_at_12"],
                             "seconds": row["seconds"], **metric})
        counts.append({"method": method, "completed": ok, "required": 600})
    pd.DataFrame(counts).to_csv(ROOT / "results/simulation_coverage.csv", index=False)
    if require_complete:
        assert all(r["completed"] == 600 for r in counts), counts
    detail = pd.DataFrame(rows)
    result = []
    for (method, scenario, rule), part in detail.groupby(["method", "scenario", "rule"]):
        row = dict(method=method, scenario=scenario, rule=rule, repetitions=len(part))
        for col in ("FDP", "power", "discoveries", "FDP_gt_0.2", "rmse_at_12"):
            vals = cinterval(part[col].astype(float))
            row.update({col + "_" + key: value for key, value in vals.items() if key != "n"})
        result.append(row)
    detail.to_csv(ROOT / "results/simulation_metrics.csv", index=False)
    pd.DataFrame(result).to_csv(ROOT / "results/simulation_summary.csv", index=False)


def summarize_prediction():
    rows = []
    for path in sorted((ROOT / "results/predictions").glob("*.json")):
        row = read(path)
        rows.append({key: row[key] for key in ("method", "evaluator", "k")} | row["test"] |
                    {"validation_rmse": row["validation"]["rmse_log"]})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/prediction_summary.csv", index=False)
    # Paired cluster bootstrap, conditional on the three observed test months.
    primary = read(ROOT / "results/development_choice.json")["primary_method"]
    intervals = []
    for evaluator in ("xgboost", "tabm"):
        for k in KS:
            reference = pd.read_csv(ROOT / "results/predictions" / f"{primary}_{evaluator}_k{k}.csv")
            sellers, idx = np.unique(reference.seller_id, return_inverse=True)
            yt = reference.log_gmv_next_month.to_numpy()
            ref_errors = (reference.prediction.to_numpy() - yt) ** 2
            ref_sums = np.bincount(idx, weights=ref_errors)
            sizes = np.bincount(idx)
            rng = np.random.default_rng(916)
            draws = rng.integers(0, len(sellers), (2000, len(sellers)))
            ref_boot = np.sqrt(ref_sums[draws].sum(1) / sizes[draws].sum(1))
            for method in COMPARATORS + ["all_features"]:
                kk = 31 if method == "all_features" else k
                frame = pd.read_csv(ROOT / "results/predictions" / f"{method}_{evaluator}_k{kk}.csv")
                assert np.array_equal(frame.row_id, reference.row_id)
                errors = (frame.prediction.to_numpy() - yt) ** 2
                sums = np.bincount(idx, weights=errors)
                boot = np.sqrt(sums[draws].sum(1) / sizes[draws].sum(1))
                difference = ref_boot - boot
                intervals.append({"primary": primary, "comparator": method, "evaluator": evaluator, "k": k,
                                  "rmse_difference": float(np.sqrt(ref_errors.mean()) - np.sqrt(errors.mean())),
                                  "ci95_low": float(np.quantile(difference, .025)),
                                  "ci95_high": float(np.quantile(difference, .975)),
                                  "familywise95_low": float(np.quantile(difference, .025 / len(COMPARATORS))),
                                  "familywise95_high": float(np.quantile(difference, 1 - .025 / len(COMPARATORS))),
                                  "multiplicity_family": "comparators within one fixed K and one evaluator",
                                  "conditional_on_observed_test_months": True})
    pd.DataFrame(intervals).to_csv(ROOT / "results/prediction_paired_intervals.csv", index=False)


def validate_core():
    rng = np.random.default_rng(81)
    x, xk = rng.normal(size=(400, 5)), rng.normal(size=(400, 5))
    y = x[:, 0] + rng.normal(size=400)
    w1 = lasso_competition(x, xk, y)
    a, b = x.copy(), xk.copy()
    a[:, 0], b[:, 0] = xk[:, 0], x[:, 0]
    w2 = lasso_competition(a, b, y)
    expected = w1.copy()
    expected[0] *= -1
    assert np.allclose(w2, expected, atol=2e-5), (w1, w2)
    assert np.isinf(threshold(np.r_[np.ones(5), np.zeros(26)], .1))
    assert threshold(np.r_[np.ones(10), np.zeros(21)], .1) == 1
    assert len(ebh(np.r_[np.repeat(20., 10), np.zeros(21)], .2)) == 10
    assert jaccard(range(12), list(range(11)) + [12]) == 11 / 13
    frame = load_data()
    assert not frame.duplicated(["seller_id", "month"]).any()
    assert set(frame.split.unique()) == {"train", "tune", "rank", "test"}
    assert frame.loc[frame.split.eq("test"), "target_month"].min() > frame.loc[
        frame.split.ne("test"), "target_month"].max()
    sample = frame.loc[frame.split.eq("train"), FEATURES].to_numpy(float)
    pre = Preprocessor(copula=True).fit(sample)
    a = pre.transform(sample[:20])
    b = pre.transform(sample[:21])[:20]
    assert np.allclose(a, b), "Test-batch-dependent CDF is forbidden"
    save_json(ROOT / "results/core_validation.json", {
        "sign_flip_max_error": float(np.abs(w2 - expected).max()), "threshold_examples": "passed",
        "ebh_example": "passed", "jaccard_example": "passed",
        "time_split_and_keys": "passed", "training_only_cdf_batch_invariance": "passed"})
    print("Core invariants passed")


def diagnose_copula():
    from deepdrk_adapter import diagnostic
    df = load_data()
    x, y, xv, yv, s, sv = split_arrays(df, ("train", "tune"), "rank")
    pre = Preprocessor(copula=True).fit(x)
    x, xv = pre.transform(x), pre.transform(xv)
    sigma = LedoitWolf().fit(x).covariance_
    sampler = GaussianSampler(X=xv.astype(float), mu=x.mean(0), Sigma=sigma, method="mvr")
    np.random.seed(11)
    generated = sampler.sample_knockoffs()
    save_json(ROOT / "results/copula_diagnostic.json", diagnostic(xv, generated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["core", "choice", "diagnostic", "summary", "progress"])
    args = parser.parse_args()
    if args.stage == "core":
        validate_core()
    elif args.stage == "choice":
        freeze_choice()
    elif args.stage == "diagnostic":
        diagnose_copula()
    elif args.stage in ("summary", "progress"):
        summarize_stability(args.stage == "summary")
        summarize_simulations(args.stage == "summary")
        if args.stage == "summary":
            summarize_prediction()
