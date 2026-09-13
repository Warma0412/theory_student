"""Audit and summarize the registered supplementary methods with matched controls."""

from run_supplement import ROOT, V7, SCPL, RESULT, P, read, selection
from common import load_data, digest_array, digest_file, save_json, jaccard, metrics
from run_experiment import discovery_metrics
from validate_and_summarize import cinterval, nogueira
from analyze import diff_stats, holm
import numpy as np
import pandas as pd


METHODS = P["methods"]
PAIRS = [(f"{model}_sko_quad_scale", f"{model}_sko_{mode}")
         for model in ("xgb", "tabm") for mode in ("ridge", "quad", "scale")]
PAIRS += [("tabm_sko_quad_scale", name) for name in ("tabm_orbit", "tabm_orbit_student", "xgb_shap")]


def audit():
    initial = read(RESULT / "initial_manifest.json")
    assert initial["protocol_sha"] == digest_file(ROOT / "protocol.json")
    assert initial["panel_sha"] == digest_file(V7 / "data_processed/seller_month_asof.parquet")
    assert initial["code_sha"] == digest_file(ROOT / "code/run_supplement.py")
    for folder, required in (("real", 111), ("simulations", 800)):
        files = list((RESULT / folder).glob("*.json"))
        assert len(files) == required, (folder, len(files))
        for path in files:
            r = read(path)
            assert r["status"] == "ok", (path, r.get("error"))
            assert set(r["methods"]) == set(METHODS)
            assert r["protocol_sha"] == initial["protocol_sha"]
            parent = read(SCPL / "results" / folder / path.name)
            fields = ("fit_row_hash", "inference_row_hash") if folder == "real" else ("x_hash", "y_hash")
            assert all(r[k] == parent[k] for k in fields)
            for method, record in r["methods"].items():
                recalculated = selection(record["score"])
                for k in ("order", "native", "native_q10"):
                    assert record[k] == recalculated[k]
                if folder == "simulations":
                    truth = np.load(SCPL / "data" / path.with_suffix(".npz").name)["truth"]
                    for rule, found in (("top12", record["order"][:12]), ("top8", record["order"][:8]),
                                        ("native", record["native"]), ("native10", record["native_q10"])):
                        assert record[rule + "_metrics"] == discovery_metrics(found, truth)
                    assert np.isfinite(record["rmse12"]) and np.isfinite(record["rmse_native"])
    test = load_data().loc[lambda d: d.split.eq("test")]
    reference = read(RESULT / "real/reference.json")["methods"]
    cache = {}
    for folder in (SCPL / "results/predictions", SCPL / "results/predictions_mvr", RESULT / "predictions"):
        for path in folder.glob("*.csv"):
            data, meta = pd.read_csv(path), read(path.with_suffix(".json"))
            assert np.array_equal(data.row_id, test.row_id)
            assert data.seller_id.tolist() == test.seller_id.tolist()
            assert np.allclose(data.log_gmv_next_month, test.log_gmv_next_month, rtol=1e-12, atol=1e-12)
            assert np.isfinite(data.prediction).all()
            if folder == RESULT / "predictions":
                assert meta["columns"] == sorted(reference[meta["method"]]["order"][:int(meta["rule"][3:])])
            for key, value in metrics(test.log_gmv_next_month.to_numpy(), data.prediction.to_numpy()).items():
                assert np.isclose(value, meta["test"][key], rtol=2e-7, atol=1e-9)
            key = (meta["evaluator"], tuple(meta["columns"]))
            if key in cache:
                assert np.array_equal(data.prediction, cache[key])
            cache[key] = data.prediction.to_numpy()
    assert len(list((RESULT / "predictions").glob("*.json"))) == 32
    save_json(RESULT / "audit.json", {"passed": True, "real_cases": 111, "simulation_cases": 800,
        "new_method_count": 8, "real_method_results": 888, "simulation_method_results": 6400,
        "prediction_outputs": 32, "new_simulation_datasets": 0, "same_input_hashes": True,
        "same_sets_same_predictions": True, "nuisance_rank_response_usage_disclosed": True,
        "source_equivalence": read(RESULT / "author_equivalence.json"),
        "protocol_sha": initial["protocol_sha"],
        "source_hashes": {p.name: digest_file(p) for p in (ROOT / "code").glob("*.py")}})


def simulations():
    rows = []
    for path in (RESULT / "simulations").glob("*.json"):
        r = read(path)
        for name, record in r["methods"].items():
            for rule in ("top12", "top8", "native", "native10"):
                rows.append({"method": name, "scene": r["scene"]["name"], "rep": r["rep"], "rule": rule,
                             **record[rule + "_metrics"],
                             "rmse": record["rmse12"] if rule == "top12" else record["rmse_native"] if rule == "native" else None})
    data = pd.concat([pd.DataFrame(rows), pd.read_csv(SCPL / "results/simulation_detail.csv")], ignore_index=True)
    data.to_csv(RESULT / "simulation_detail.csv", index=False)
    summary, paired = [], []
    for (scene, method, rule), sub in data.groupby(["scene", "method", "rule"]):
        row = {"scene": scene, "method": method, "rule": rule}
        for metric in ("FDP", "power", "size", "rmse"):
            vals = sub[metric].dropna()
            if len(vals):
                row.update({f"{metric}_{k}": v for k, v in cinterval(vals).items()})
        summary.append(row)
    for (scene, rule), sub in data.groupby(["scene", "rule"]):
        for metric in ("FDP", "power", "rmse"):
            matrix = sub.pivot(index="rep", columns="method", values=metric)
            family = []
            for a, b in PAIRS:
                if a in matrix and b in matrix:
                    diff = (matrix[a] - matrix[b]).dropna()
                    if len(diff) == 100:
                        family.append({"scene": scene, "rule": rule, "metric": metric,
                                       "candidate": a, "comparator": b, **diff_stats(diff)})
            for row, pvalue in zip(family, holm([r["p_two_sided"] for r in family])):
                paired.append(row | {"holm_p": pvalue})
    pd.DataFrame(summary).to_csv(RESULT / "simulation_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(RESULT / "simulation_paired.csv", index=False)


def stability():
    reference = read(RESULT / "real/reference.json")["methods"]
    specs = read(V7 / "data_processed/sampling_manifest.json")
    rows = []
    for spec in specs:
        methods = read(RESULT / "real" / f"{spec['run']}.json")["methods"]
        for name, record in methods.items():
            rows.append({"method": name, "fraction": spec["fraction"], "group": spec["group"],
                         "run": spec["run"], "half": spec["half"],
                         "J": jaccard(record["order"][:12], reference[name]["order"][:12]),
                         "set": __import__("json").dumps(record["order"][:12]),
                         "native_size": len(record["native"])})
    data = pd.concat([pd.DataFrame(rows), pd.read_csv(SCPL / "results/stability_detail.csv")], ignore_index=True)
    summary, complements, paired = [], [], []
    for (name, fraction), part in data.groupby(["method", "fraction"]):
        sets = [__import__("json").loads(v) for v in part["set"]]
        summary.append({"method": name, "fraction": fraction, **cinterval(part.groupby("group").J.mean()),
                        "nogueira": nogueira(sets)})
        if fraction == .5:
            for group, sub in part.groupby("group"):
                a, b = [__import__("json").loads(v) for v in sub["set"]]
                complements.append({"method": name, "group": group, "J": jaccard(a, b)})
    matrix = data.loc[data.fraction.eq(.5)].groupby(["method", "group"]).J.mean().unstack(0)
    for a, b in PAIRS:
        paired.append({"candidate": a, "comparator": b, **diff_stats(matrix[a] - matrix[b])})
    data.to_csv(RESULT / "stability_detail.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULT / "stability_summary.csv", index=False)
    pd.DataFrame(complements).to_csv(RESULT / "stability_complementary.csv", index=False)
    pd.DataFrame(paired).to_csv(RESULT / "stability_paired.csv", index=False)
    data.loc[data.method.isin(METHODS), ["method", "fraction", "group", "run", "native_size"]].to_csv(
        RESULT / "native_sizes.csv", index=False)
    algorithm = []
    for path in (RESULT / "real").glob("seedonly_*.json"):
        for name, record in read(path)["methods"].items():
            algorithm.append({"method": name, "run": path.stem,
                              "J": jaccard(record["order"][:12], reference[name]["order"][:12])})
    pd.DataFrame(algorithm).to_csv(RESULT / "algorithm_randomness.csv", index=False)


def predictions():
    rows = []
    for folder in (RESULT / "predictions", SCPL / "results/predictions"):
        for path in folder.glob("*.json"):
            r = read(path)
            rows.append({"method": r["method"], "evaluator": r["evaluator"], "rule": r["rule"], **r["test"]})
    pd.DataFrame(rows).to_csv(RESULT / "prediction_summary.csv", index=False)
    paired = []
    for evaluator in ("xgboost", "tabm"):
        for k in (12, 8):
            rule = f"top{k}"
            base = pd.read_csv(SCPL / "results/predictions" / f"xgb_shap_{evaluator}_{rule}.csv")
            ids, index = np.unique(base.seller_id, return_inverse=True)
            sizes = np.bincount(index)
            draws = np.random.default_rng(1735).integers(0, len(ids), (5000, len(ids)))
            boot, point = {}, {}
            for name in set(v for pair in PAIRS + [(m, "xgb_shap") for m in METHODS] for v in pair):
                folder = RESULT / "predictions" if name in METHODS else SCPL / "results/predictions"
                data = pd.read_csv(folder / f"{name}_{evaluator}_{rule}.csv")
                error = (data.prediction.to_numpy() - base.log_gmv_next_month.to_numpy())**2
                total = np.bincount(index, weights=error)
                boot[name] = np.sqrt(total[draws].sum(1) / sizes[draws].sum(1))
                point[name] = np.sqrt(error.mean())
            for a, b in dict.fromkeys(PAIRS + [(m, "xgb_shap") for m in METHODS]):
                delta = boot[a] - boot[b]
                paired.append({"candidate": a, "comparator": b, "evaluator": evaluator, "rule": rule,
                               "difference": point[a] - point[b], "ci_low": np.quantile(delta, .025),
                               "ci_high": np.quantile(delta, .975), "exploratory_pointwise_interval": True})
    pd.DataFrame(paired).to_csv(RESULT / "prediction_paired.csv", index=False)


if __name__ == "__main__":
    audit()
    simulations()
    stability()
    predictions()
    print("Supplementary audit and summaries complete")
