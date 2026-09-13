"""Audit complete coverage and report paired evidence without selective ranking."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

from run_study import STUDY, V7, RESULT, P, METHODS, FAMILIES, read
from common import FEATURES, digest_array, digest_file, load_data, save_json, jaccard
from validate_and_summarize import cinterval, nogueira, LABELS as BASE_LABELS
import numpy as np
import pandas as pd

BASES = ["copula_mvr", "elastic_net", "stability_selection", "shadow_trees",
         "xgboost_shap", "deep_lasso", "tabm", "vtfs"]
ALL = BASES + METHODS
LABELS = BASE_LABELS | {
    "dl_jacobian": "Deep Lasso-Jacobian",
    "dl_jacobian_lossrank": "Jacobian训练+原排名",
    "dl_score_only": "原Deep Lasso+Jacobian排名",
    "tabm_budget_gate": "TabM-预算门",
    "ep_regression": "EntryPrune-回归适配",
    "ep_refresh": "EntryPrune-刷新",
    "ep_dispersion": "EntryPrune-环境波动惩罚",
    "vtfs_cardinality": "VTFS-固定K",
    "vtfs_pairrank": "VTFS-固定K+成对损失",
    "random_k_search": "等预算随机子集搜索",
}
PARENTS = {
    "dl_jacobian": "deep_lasso", "dl_jacobian_lossrank": "deep_lasso",
    "dl_score_only": "deep_lasso", "tabm_budget_gate": "tabm",
    "ep_refresh": "ep_regression", "ep_dispersion": "ep_regression",
    "vtfs_cardinality": "vtfs", "vtfs_pairrank": "vtfs_cardinality",
    "random_k_search": "vtfs",
}
PAIRS = list(PARENTS.items()) + [(m, "xgboost_shap") for m in METHODS] + [
    ("ep_dispersion", "ep_refresh"), ("vtfs_pairrank", "random_k_search"),
    ("vtfs_cardinality", "random_k_search"), ("dl_jacobian", "dl_score_only"),
    ("dl_jacobian", "dl_jacobian_lossrank"),
]
SCENES = P["evaluation"]["simulation_scenarios"]


def source(method):
    return (V7 if method in BASES or method == "all_features" else STUDY) / "results"


def runfile(method, case):
    if case == "reference_k8" and method in BASES:
        case = "reference"
    return source(method) / "runs" / method / f"{case}.json"


def prediction_file(method, evaluator, k, suffix=".json"):
    return RESULT / "canonical_predictions" / f"{method}_{evaluator}_k{k}{suffix}"


def coverage():
    rows, failures = [], []
    for method in METHODS:
        for kind, need in (("runs", 112), ("simulation_runs", 600)):
            found, success = 0, 0
            for p in (RESULT / kind / method).glob("*.json"):
                r = read(p)
                found += 1
                success += r.get("status") == "ok"
                if r.get("status") != "ok":
                    failures.append({"method": method, "file": str(p), "error": r.get("error")})
            rows.append({"method": method, "stage": kind, "found": found, "successful": success, "required": need})
    table = pd.DataFrame(rows)
    table.to_csv(RESULT / "coverage.csv", index=False)
    save_json(RESULT / "failures.json", failures)
    print(table.to_string(index=False))
    return table


def audit():
    table = coverage()
    assert (table.successful == table.required).all(), "Incomplete cases or failed fits"
    frame = load_data()
    manifest = read(V7 / "data_processed/sampling_manifest.json")
    assert len(manifest) == 100
    protocol_hash = digest_file(STUDY / "protocol.json")
    start = read(RESULT / "source_manifest.json")
    assert start["protocol_hash"] == protocol_hash
    assert start["panel_hash"] == digest_file(V7 / "data_processed/seller_month_asof.parquet")
    assert start["sampling_hash"] == digest_file(V7 / "data_processed/sampling_manifest.json")
    for name in ("models.py", "common.py", "run_experiments.py", "vtfs_adapter.py", "build_panel.py"):
        assert start["parent_code_hashes"][name] == digest_file(V7 / "code" / name)
    for method in BASES:
        for spec in manifest:
            row = read(runfile(method, spec["run"]))
            subset = frame.loc[frame.seller_id.isin(spec["sellers"])]
            assert row["fit_row_hash"] == digest_array(subset.loc[
                subset.split.isin(["train", "tune"]), "row_id"].to_numpy())
            assert row["rank_row_hash"] == digest_array(subset.loc[subset.split.eq("rank"), "row_id"].to_numpy())
    for method in METHODS:
        for spec in manifest:
            row = read(runfile(method, spec["run"]))
            subset = frame.loc[frame.seller_id.isin(spec["sellers"])]
            for splits, key in ((["train", "tune"], "fit_row_hash"), (["rank"], "rank_row_hash")):
                assert row[key] == digest_array(subset.loc[subset.split.isin(splits), "row_id"].to_numpy())
            assert row["test_rows_read_by_selector"] == 0 and row["protocol_hash"] == protocol_hash
            assert sorted(row["order"]) == list(range(31))
        for k in (8, 12):
            case = "reference" if k == 12 else "reference_k8"
            assert read(runfile(method, case))["k"] == k
            for evaluator in ("xgboost", "tabm"):
                path = RESULT / "predictions" / f"{method}_{evaluator}_k{k}.csv"
                data = pd.read_csv(path)
                meta = read(path.with_suffix(".json"))
                assert len(data) == 3413 and np.isfinite(data.prediction).all()
                assert np.array_equal(data.row_id, frame.loc[frame.split.eq("test"), "row_id"])
                assert meta["test_labels_not_used_for_tuning"]
                expected = [FEATURES[j] for j in read(runfile(method, case))["order"][:k]]
                assert meta["features"] == expected
    prediction_cache = {}
    for method in ALL + ["all_features"]:
        for k in ([31] if method == "all_features" else [8, 12]):
            expected = list(range(31)) if k == 31 else sorted(
                read(runfile(method, "reference" if k == 12 else "reference_k8"))["order"][:k])
            for ev in ("xgboost", "tabm"):
                meta = read(prediction_file(method, ev, k))
                data = pd.read_csv(prediction_file(method, ev, k, ".csv"))
                assert meta["features"] == [FEATURES[j] for j in expected]
                assert np.array_equal(data.row_id, frame.loc[frame.split.eq("test"), "row_id"])
                key = (tuple(expected), ev)
                if key in prediction_cache:
                    assert np.array_equal(data.prediction, prediction_cache[key])
                prediction_cache[key] = data.prediction.to_numpy()
    for scene in SCENES:
        for rep in range(100):
            case = f"{scene}_{rep:03d}"
            raw = dict(np.load(V7 / "data_processed/simulations" / f"{case}.npz"))
            evaluation = read(RESULT / "canonical_simulation_evaluation" / f"{case}.json")
            assert evaluation["data_hash"] == digest_array(raw["x"])
            truth = set(map(int, raw["truth"]))
            for method in BASES:
                base = read(source(method) / "simulation_runs" / method / f"{case}.json")
                assert base["status"] == "ok" and base["data_hash"] == digest_array(raw["x"])
                assert np.array_equal(base["truth"], raw["truth"])
            for method in METHODS:
                row = read(RESULT / "simulation_runs" / method / f"{case}.json")
                assert row["data_hash"] == digest_array(raw["x"]) and row["response_hash"] == digest_array(raw["y"])
                selected = set(row["order"][:12])
                assert len(selected) == 12
                assert np.isclose(row["FDP"], len(selected - truth) / 12)
                assert np.isclose(row["power"], len(selected & truth) / len(truth))
                assert row["protocol_hash"] == protocol_hash
                assert evaluation["methods"][method]["columns"] == sorted(selected)
                if method in ("vtfs_cardinality", "vtfs_pairrank", "random_k_search"):
                    assert row["metadata"]["queries"] == 128
    save_json(RESULT / "audit.json", {
        "passed": True, "new_primary_real_cases": len(METHODS) * 111,
        "new_secondary_reference_cases": len(METHODS), "new_simulation_cases": len(METHODS) * 600,
        "new_prediction_files": len(METHODS) * 4,
        "canonical_prediction_files_including_baselines": len(ALL) * 4 + 2,
        "canonical_simulation_evaluation_cases": 600,
        "identical_selected_sets_have_identical_real_predictions": True,
        "same_seller_rows_and_simulation_inputs": True, "fixed_cardinality_and_metric_recalculation": True,
        "protocol_hash": protocol_hash, "current_code_hashes": {p.name: digest_file(p) for p in (STUDY / "code").glob("*.py")},
        "historical_baselines_read_not_retrained": BASES,
        "unchanged_parent_model_code_panel_sampling_manifest": True,
        "post_v7_exploratory_not_an_external_validation": True,
    })


def summarize_stability():
    specs = read(V7 / "data_processed/sampling_manifest.json")
    rows, seedrows = [], []
    for method in ALL:
        full = read(runfile(method, "reference"))
        for spec in specs:
            r = read(runfile(method, spec["run"]))
            rows.append({"method": method, "fraction": spec["fraction"], "group": spec["group"],
                         "half": spec["half"], "run": spec["run"],
                         "J": jaccard(full["order"][:12], r["order"][:12]),
                         "features": json.dumps(r["order"][:12])})
        for i in range(10):
            r = read(runfile(method, f"seedonly_{i:02d}"))
            seedrows.append({"method": method, "batch": i, "J": jaccard(full["order"][:12], r["order"][:12])})
    details = pd.DataFrame(rows)
    summaries, complement = [], []
    for (m, frac), df in details.groupby(["method", "fraction"]):
        group = df.groupby("group").J.mean()
        summaries.append({"method": m, "fraction": frac, **cinterval(group),
                          "nogueira": nogueira([json.loads(v) for v in df.features])})
        if frac == .5:
            for group, halves in df.groupby("group"):
                a, b = halves.sort_values("half").features.map(json.loads)
                complement.append({"method": m, "group": group, "J": jaccard(a, b)})
    pair_rows = []
    comp = pd.DataFrame(complement)
    pivot = comp.pivot(index="group", columns="method", values="J")
    ref = details.loc[details.fraction.eq(.5)].groupby(["method", "group"]).J.mean().unstack(0)
    for a, b in PAIRS:
        for metric, data in (("complementary_J", pivot), ("reference_J", ref)):
            values = data[a] - data[b]
            ci = cinterval(values)
            # Bonferroni t intervals, limited to this prespecified family.
            from scipy.stats import t
            width = t.ppf(1 - .025 / len(PAIRS), len(values) - 1) * ci["se"]
            pair_rows.append({"candidate": a, "comparator": b, "metric": metric, **ci,
                              "bonf_low": ci["mean"] - width, "bonf_high": ci["mean"] + width})
    pd.DataFrame(summaries).to_csv(RESULT / "stability_summary.csv", index=False)
    details.to_csv(RESULT / "stability_runs.csv", index=False)
    pd.DataFrame(seedrows).to_csv(RESULT / "seed_stability.csv", index=False)
    comp.to_csv(RESULT / "complementary_J.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(RESULT / "stability_paired.csv", index=False)


def summarize_simulation():
    rows, diagnostics = [], []
    for method in ALL:
        for scene in SCENES:
            for rep in range(100):
                row = read(source(method) / "simulation_runs" / method / f"{scene}_{rep:03d}.json")
                assert row["status"] == "ok"
                if method in BASES:
                    record = next(m for m in row["metrics"] if m["rule"] == "top12")
                    fdp, power, rmse = record["FDP"], record["power"], row["rmse_at_12"]
                else:
                    fdp, power, rmse = row["FDP"], row["power"], row["rmse"]
                    if method.startswith("vtfs_"):
                        meta = row["metadata"]
                        diagnostics.append({"method": method, "scene": scene, "rep": rep,
                            "neural_winner": meta["winner_neural_candidate"],
                            "decoder_gain": meta["decoder_gain_over_queried_fallbacks"]})
                rmse = read(RESULT / "canonical_simulation_evaluation" /
                            f"{scene}_{rep:03d}.json")["methods"][method]["rmse"]
                rows.append({"method": method, "scene": scene, "rep": rep, "FDP": fdp, "power": power, "rmse": rmse})
    data = pd.DataFrame(rows)
    summary, paired = [], []
    for (method, scene), df in data.groupby(["method", "scene"]):
        entry = {"method": method, "scene": scene}
        for metric in ("FDP", "power", "rmse"):
            entry.update({f"{metric}_{k}": v for k, v in cinterval(df[metric]).items()})
        summary.append(entry)
    for scene in SCENES:
        part = data.loc[data.scene.eq(scene)]
        for metric in ("FDP", "power", "rmse"):
            matrix = part.pivot(index="rep", columns="method", values=metric)
            for a, b in PAIRS:
                paired.append({"candidate": a, "comparator": b, "scene": scene, "metric": metric,
                               **cinterval(matrix[a] - matrix[b])})
    data.to_csv(RESULT / "simulation_detail.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULT / "simulation_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(RESULT / "simulation_paired.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULT / "vtfs_decoder_diagnostic.csv", index=False)


def predictions():
    result, differences, redundancy = [], [], []
    training = load_data().loc[lambda d: d.split.isin(["train", "tune"])]
    corr = training[FEATURES].corr(method="spearman").abs()
    for method in ALL + ["all_features"]:
        for k in ([31] if method == "all_features" else [8, 12]):
            meta_path = prediction_file(method, "xgboost", k)
            features = read(meta_path)["features"]
            values = [corr.loc[a, b] for a, b in combinations(features, 2)]
            redundancy.append({"method": method, "k": k,
                               "mean_abs_corr": float(np.mean(values)),
                               "high_corr_pairs": int(np.sum(np.asarray(values) >= .8)),
                               "fit_window_only": True})
            for evaluator in ("xgboost", "tabm"):
                meta = read(prediction_file(method, evaluator, k))
                frame = pd.read_csv(prediction_file(method, evaluator, k, ".csv"))
                monthly = [float(np.mean((d.prediction - d.log_gmv_next_month)**2)**.5)
                           for _, d in frame.groupby("target_month")]
                result.append({"method": method, "evaluator": evaluator, "k": k, **meta["test"],
                               "worst_month_rmse": max(monthly),
                               "validation_rmse": meta["validation"]["rmse_log"]})
    for k in (8, 12):
        for evaluator in ("xgboost", "tabm"):
            ref = pd.read_csv(prediction_file("xgboost_shap", evaluator, k, ".csv"))
            ids, idx = np.unique(ref.seller_id, return_inverse=True)
            sizes = np.bincount(idx)
            draws = np.random.default_rng(916).integers(0, len(ids), (5000, len(ids)))
            boot, points = {}, {}
            for m in ALL:
                frame = pd.read_csv(prediction_file(m, evaluator, k, ".csv"))
                assert np.array_equal(frame.row_id, ref.row_id)
                errors = (frame.prediction.to_numpy() - ref.log_gmv_next_month.to_numpy())**2
                sums = np.bincount(idx, weights=errors)
                boot[m] = np.sqrt(sums[draws].sum(1) / sizes[draws].sum(1))
                points[m] = float(np.sqrt(errors.mean()))
            for a, b in PAIRS:
                diff = boot[a] - boot[b]
                ratio = boot[a] / boot[b] - 1
                differences.append({"candidate": a, "comparator": b, "evaluator": evaluator, "k": k,
                    "rmse_diff": points[a] - points[b], "relative_diff": points[a] / points[b] - 1,
                    "ci_low": float(np.quantile(diff, .025)), "ci_high": float(np.quantile(diff, .975)),
                    "ratio_ci_low": float(np.quantile(ratio, .025)), "ratio_ci_high": float(np.quantile(ratio, .975)),
                    "bonf_low": float(np.quantile(diff, .025 / (2 * len(PAIRS)))),
                    "bonf_high": float(np.quantile(diff, 1 - .025 / (2 * len(PAIRS)))),
                    "family": "all prespecified comparisons x two evaluators at this K",
                    "conditional_on_three_months": True, "exploratory": True})
    pd.DataFrame(result).to_csv(RESULT / "prediction_summary.csv", index=False)
    pd.DataFrame(differences).to_csv(RESULT / "prediction_paired.csv", index=False)
    pd.DataFrame(redundancy).to_csv(RESULT / "redundancy.csv", index=False)


def mechanism_diagnostics():
    frame = pd.read_csv(V7 / "results/feature_dictionary.csv")
    specs = read(V7 / "data_processed/sampling_manifest.json")[:60]
    rows, entry_churn = [], []
    for method in ALL:
        frequency = np.zeros(31)
        full = set(read(runfile(method, "reference"))["order"][:12])
        for spec in specs:
            frequency[read(runfile(method, spec["run"]))["order"][:12]] += 1
        for j, feature in enumerate(FEATURES):
            rows.append({"method": method, "feature": feature, "label_zh": frame.label_zh.iloc[j],
                         "full_selected": j in full, "half_sample_frequency": frequency[j] / 60})
        if method.startswith("ep_"):
            for spec in specs:
                rec = read(runfile(method, spec["run"]))
                for fit in rec["metadata"]["fits"]:
                    supports = [r[2] for r in fit["history"]]
                    changes = [1 - jaccard(a, b) for a, b in zip(supports, supports[1:])]
                    entry_churn.append({"method": method, "run": spec["run"], "seed": fit["seed"],
                                        "mean_step_jaccard_distance": np.mean(changes),
                                        "best_update": fit["best_update"]})
    pd.DataFrame(rows).to_csv(RESULT / "feature_frequencies.csv", index=False)
    pd.DataFrame(entry_churn).to_csv(RESULT / "entry_training_churn.csv", index=False)


def tradeoff_table():
    prediction = pd.read_csv(RESULT / "prediction_paired.csv")
    stability = pd.read_csv(RESULT / "stability_paired.csv")
    simulations = pd.read_csv(RESULT / "simulation_paired.csv")
    rows = []
    for a, b in PAIRS:
        entry = {"candidate": a, "comparator": b}
        for ev in ("xgboost", "tabm"):
            r = prediction.loc[prediction.candidate.eq(a) & prediction.comparator.eq(b) &
                               prediction.evaluator.eq(ev) & prediction.k.eq(12)].iloc[0]
            entry.update({f"{ev}_rmse_diff": r.rmse_diff, f"{ev}_ratio_ci_high": r.ratio_ci_high,
                          f"{ev}_pointwise_within_1pct": bool(r.ratio_ci_high <= .01)})
        for metric in ("reference_J", "complementary_J"):
            r = stability.loc[stability.candidate.eq(a) & stability.comparator.eq(b) &
                              stability.metric.eq(metric)].iloc[0]
            entry.update({f"{metric}_diff": r["mean"], f"{metric}_ci_low": r.ci_low,
                          f"{metric}_bonf_low": r.bonf_low})
        for metric in ("FDP", "power"):
            part = simulations.loc[simulations.candidate.eq(a) & simulations.comparator.eq(b) &
                                   simulations.metric.eq(metric)]
            assert len(part) == 6
            mean, se = part["mean"].mean(), np.sqrt((part.se**2).sum()) / 6
            entry.update({f"mean_scene_{metric}_diff": mean, f"mean_scene_{metric}_ci_low": mean - 1.96 * se,
                          f"mean_scene_{metric}_ci_high": mean + 1.96 * se})
        rows.append(entry)
    pd.DataFrame(rows).to_csv(RESULT / "tradeoff_evidence.csv", index=False)


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *[line(row) for row in rows]])


def report():
    pred = pd.read_csv(RESULT / "prediction_summary.csv")
    stable = pd.read_csv(RESULT / "stability_summary.csv")
    sim = pd.read_csv(RESULT / "simulation_summary.csv")
    paired = pd.read_csv(RESULT / "prediction_paired.csv")
    redundancy = pd.read_csv(RESULT / "redundancy.csv")
    rows = []
    for method in ALL:
        part = pred.loc[pred.method.eq(method) & pred.k.eq(12)].set_index("evaluator")
        ss = stable.loc[stable.method.eq(method) & stable.fraction.eq(.5)].iloc[0]
        sm = sim.loc[sim.method.eq(method)]
        rows.append([LABELS[method], f"{part.loc['xgboost','rmse_log']:.4f}",
                     f"{part.loc['tabm','rmse_log']:.4f}", f"{ss['mean']:.4f}",
                     f"{ss.nogueira:.4f}", f"{sm.FDP_mean.mean():.4f}", f"{sm.power_mean.mean():.4f}"])
    summary_table = table(["方法", "XGB-RMSE", "TabM-RMSE", "50% Jaccard", "Nogueira", "FDP@12", "Power@12"], rows)
    rows = []
    for a, b in PAIRS[:len(PARENTS)]:
        for evaluator in ("xgboost", "tabm"):
            r = paired.loc[paired.candidate.eq(a) & paired.comparator.eq(b) &
                           paired.evaluator.eq(evaluator) & paired.k.eq(12)].iloc[0]
            rows.append([LABELS[a], LABELS[b], evaluator, f"{r.rmse_diff:+.4f}",
                         f"[{r.ci_low:.4f}, {r.ci_high:.4f}]", f"[{r.bonf_low:.4f}, {r.bonf_high:.4f}]"])
    diff_table = table(["改动", "父/对照", "评价器", "RMSE差", "点区间95%", "族内校正区间"], rows)
    scene_tables = []
    for scene in SCENES:
        rows = []
        for method in ALL:
            r = sim.loc[sim.method.eq(method) & sim.scene.eq(scene)].iloc[0]
            rows.append([LABELS[method], f"{r.FDP_mean:.4f}", f"{r.FDP_se:.4f}",
                         f"{r.power_mean:.4f}", f"{r.rmse_mean:.4f}"])
        scene_tables += [f"### {scene}", table(["方法", "平均FDP", "FDP MCSE", "Power", "XGB-RMSE"], rows)]
    config_table = table(["模型家族", "选定参数", "选择依据"], [
        [family, read(RESULT / "configs" / f"{family}.json").get("value"), "1月排序；2月XGB误差" if family != "subset" else "协议固定"]
        for family in FAMILIES])
    diagnostic = pd.read_csv(RESULT / "vtfs_decoder_diagnostic.csv")
    diagrows = []
    for m, d in diagnostic.groupby("method"):
        ref = read(runfile(m, "reference"))["metadata"]
        diagrows.append([LABELS[m], f"{d.neural_winner.mean():.3f}",
                         f"{d.decoder_gain.mean():.6f}", str(ref["winner_neural_candidate"])])
    diagtable = table(["方法", "600模拟神经候选胜出比例", "模拟平均开发效用增益", "真实完整样本神经候选胜出"], diagrows)
    scene_text = "\n\n".join(scene_tables)
    def sj(method):
        return float(stable.loc[stable.method.eq(method) & stable.fraction.eq(.5), "mean"].iloc[0])
    def sm(method, metric):
        return float(sim.loc[sim.method.eq(method), metric + "_mean"].mean())
    conclusions = f"""
1. **Deep Lasso-Jacobian：有局部价值，尚非全面提升。** 相对完整名单的半样本Jaccard从原版{sj('deep_lasso'):.4f}升至{sj('dl_jacobian'):.4f}，但相对父方法的互补半样本增量区间跨0，参考Jaccard增量经族内校正后也跨0。相对SHAP的两种Jaccard差异较明确；其平均FDP为{sm('dl_jacobian','FDP'):.4f}、Power为{sm('dl_jacobian','power'):.4f}，仍弱于SHAP的{sm('xgboost_shap','FDP'):.4f}和{sm('xgboost_shap','power'):.4f}。适合跟进稳定性取舍，不支持全面优胜或新定理。
2. **TabM预算门：不支持综合改善。** 参考Jaccard从{sj('tabm'):.4f}升至{sj('tabm_budget_gate'):.4f}，但互补一致性和Nogueira没有同步提高，模拟FDP从{sm('tabm','FDP'):.4f}升至{sm('tabm_budget_gate','FDP'):.4f}。不能只选参考Jaccard作为成功标准。
3. **EntryPrune刷新与波动惩罚：本实现不宜作为提升主线。** 波动版相对回归父实现的参考Jaccard从{sj('ep_regression'):.4f}降至{sj('ep_dispersion'):.4f}，Power从{sm('ep_regression','power'):.4f}降至{sm('ep_dispersion','power'):.4f}。即使一个评价器的误差点估计更低，也不能掩盖选择质量恶化。刷新涉及分数与动量两项变化，不能将失败单独归因于其中一项。
4. **VTFS：预算匹配有模拟收益，成对损失独立证据较弱。** 原适配、固定K、固定K加成对损失的平均FDP依次为{sm('vtfs','FDP'):.4f}、{sm('vtfs_cardinality','FDP'):.4f}、{sm('vtfs_pairrank','FDP'):.4f}，Power依次为{sm('vtfs','power'):.4f}、{sm('vtfs_cardinality','power'):.4f}、{sm('vtfs_pairrank','power'):.4f}。主要收益来自预算修正；成对损失较固定K版本仅小幅增量。二者真实数据预测与稳定性尚未形成优势，修复旧适配不应包装成原创模型成功。

结论：没有发现同时在预测、两类样本稳定性与模拟检出方面全面超过强基线的新算法。计算完成和某项指标变好，都不能自动证明论文原创性。
"""
    text = f"""# 近期模型算法改动的完整配对实验

## 1. 结论性质与范围

本研究是已观察V7结果之后的探索，不是外部预注册验证。新改动均完整执行，不能把一次改动的局部收益说成世界首创或跨数据集普遍优势。V7原论文、CSV、模型和失败记录没有覆盖。

新增十个方法/消融配置，各完成111个主预算真实案例、一个8项参考案例及600个模拟案例，合计1,120个新真实选择案例、6,000个新模拟案例、40组初始预测文件。旧八方法的选择结果作为历史基线读取，不冒充选择器重新训练。所有18种方法与全维基线随后共同按规范列顺序重评，形成74组预测文件。完整原始表构建的13,754行卖家月面板未抽小；外层抽样是评价的一部分。

## 2. 四条模型改动

1. **Deep Lasso-Jacobian**：原梯度分数同时乘有残差，容易使误差大的样本支配重要性。改为预测函数Jacobian的按列RMS稀疏目标，并用相同Jacobian排序。另做“只改排序”和“改训练但保留原排名”两项消融。导数正则已有长期文献，不能称为首次提出。
2. **TabM-预算门**：在官方8成员TabM前加入共享的精确K硬门，反向传播使用约束sigmoid软门的直通梯度；15轮全变量预热后开始预算训练。解决训练看31项、最终只留K项的不一致。输入门与直通估计均有先例，本实现是局部预算约束适配。
3. **EntryPrune-刷新/环境波动惩罚**：原发布代码记住候选历史最高进入分数。刷新版用三个训练时间块的当前梯度均分并重置再生列Adam动量，波动版再扣除环境间标准差。刷新版和波动版共享除波动项之外的改动；这不是具有覆盖保证的置信下界。回归输出和对齐最佳检查点是共同适配，不列为创新。
4. **VTFS-固定K/成对损失**：固定K语料与定长唯一token解码消除不同长度和任意前缀造成的预算偏差；新增按效用差加权的成对logistic排序损失。用同样128次子集查询的随机搜索检验Transformer是否带来额外收益。固定K修复本身不冒称原论文的新发现。

官方组件：Deep Lasso官方正则与V7 MLP；TabM 0.0.3；EntryPrune锁定commit；VTFS官方编码/解码骨干（仍保留epsilon=1）。所有路径与摘要见source_manifest.json。

## 3. 公平条件与限制

各方法使用同一30组互补50%卖家、20个70%、20个80%样本，另有10组固定数据种子变化。半样本区间以30组计算。拟合与排序行摘要逐项核对，选择器不读测试X或Y。模型新增参数在2017年训练、2018年1月早停/排序、2月评价后冻结；最终名单以训练加调参期拟合、3至4月排序。

{config_table}

主比较K=12；K=8仅完整参考与预测敏感性，门控、EntryPrune和VTFS独立按8项训练。XGBoost和TabM评价器配置、三个训练种子与全开发窗口重拟合对所有名单一致。方法自身计算量并不完全相等。冻结的V7架构参数、局部有限搜索不代表每一种模型的充分调参性能上界。

所有新方法只输出排名/固定预算名单，不提供原生FDR保证。FDP与Power在模拟真值下计算；固定K和真信号数时二者一一对应，不能当作两份独立成功证据。六场景各100次等权汇总只用于总览，逐场景表必须同时阅读。

执行审计发现两项Deep Lasso排名消融的完整样本Top-12与原版相同，但不同列顺序会影响随机列抽样和网络初始化。为消除伪差异，本报告最终预测统一按特征编号排序后重算，初始预测保留。该修订不改变选择名单、参数、FDP、Power或稳定性。相同集合在相同评价器中必须给出相同预测，并已纳入断言。参数仍沿用原先冻结的开发期选择，不因重评结果调整。

模拟下游预测也统一使用标准化表示和规范列顺序，保留各方法原名单。这同时纠正了历史Copula模拟评价器使用秩高斯表示的问题；新评价保存在canonical_simulation_evaluation，不覆盖V7，也不计为新增选择算法案例。

## 4. 主预算总览

{summary_table}

RMSE越低越好，Jaccard/Nogueira/Power越高越好，FDP越低越好。Jaccard是相对各自完整开发名单；算法种子稳定性另存，不与样本稳定性混用。平均低FDP不代表实际Olist错误率已控。

### 各改动的实际判断

{conclusions}

## 5. 对父方法的配对差异

下表差为“改动减对照”，RMSE差负数有利于改动。卖家簇Bootstrap 5,000次，以三个已观察测试月份为条件。校正区间覆盖预先列出的比较对与两个评价器；不能解读为独立外部确认。尾部分位数仍有Monte Carlo误差。

{diff_table}

不能因为点区间跨0就宣称等价；1%容忍度只是预设的工程参考。即便优于较弱父方法，仍需与XGBoost-SHAP、影子树等强基线比较，并查看检出率及稳定性代价。

预测Bootstrap条件于当前已拟合的三种子集成，不包含重新训练、重调参数或换到新时间环境的不确定性。TabM门控试验同时改变了预热和预算训练日程，因此即使发现收益，也不能在缺少额外日程消融时把全部效果只归功于门本身。

## 6. 逐场景模拟

{scene_text}

## 7. 组件是否真正贡献

{diagtable}

“神经候选胜出”只是该次有限搜索中的来源标签，不自动证明相对全局随机搜索或样本外目标有优势。Deep Lasso的训练/排名交叉消融、EntryPrune扣波动前后比较、VTFS成对损失和等查询随机对照均保留于配对统计文件。

## 8. 复现与剩余研究

本轮仍然是一个电商数据集和预设模拟，未声称第二数据集验证。已有测试期被查看过，报告有利结果不能消除选择偏差。正式论文本体采纳某条改动之前，需要独立时间或数据集、相关先行研究对比及足够的理论/机制说明，不能以“跑赢”替代原创性判断。

结果文件：coverage.csv、audit.json、prediction_summary.csv、prediction_paired.csv、stability_summary.csv、stability_paired.csv、simulation_summary.csv、simulation_paired.csv、vtfs_decoder_diagnostic.csv。逐样本结果及参考模型权重均保留。
"""
    (STUDY / "实验报告.md").write_text(text)
    pd.DataFrame([{
        "method": m, "label": LABELS[m],
        "rmse_xgb": pred.loc[pred.method.eq(m) & pred.k.eq(12) & pred.evaluator.eq("xgboost"), "rmse_log"].iloc[0],
        "rmse_tabm": pred.loc[pred.method.eq(m) & pred.k.eq(12) & pred.evaluator.eq("tabm"), "rmse_log"].iloc[0],
        "jaccard_half": stable.loc[stable.method.eq(m) & stable.fraction.eq(.5), "mean"].iloc[0],
        "FDP12_mean_scenes": sim.loc[sim.method.eq(m), "FDP_mean"].mean(),
        "Power12_mean_scenes": sim.loc[sim.method.eq(m), "power_mean"].mean(),
    } for m in ALL]).to_csv(RESULT / "comparison_overview.csv", index=False)
    print(summary_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["progress", "audit", "summary", "report"])
    args = parser.parse_args()
    if args.stage == "progress":
        coverage()
    elif args.stage == "audit":
        audit()
    elif args.stage == "summary":
        audit()
        summarize_stability()
        summarize_simulation()
        predictions()
        mechanism_diagnostics()
        tradeoff_table()
        report()
    else:
        report()
