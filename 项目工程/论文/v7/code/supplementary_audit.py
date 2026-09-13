"""Post-protocol diagnostics; never alter the frozen primary comparison."""

from __future__ import annotations

import argparse
import json
import time

from common import ROOT, FEATURES, SEEDS, digest_array, digest_file, save_json, jaccard, load_data
from models import run_selector
from run_experiments import transformed, config_path
from validate_and_summarize import COMPARATORS, read, cinterval
import numpy as np
import pandas as pd


def deep_lasso_ablation(shard=0, shards=1):
    df = load_data()
    dev = df.loc[df.split.ne("test")]
    specs = [{"run": "reference", "sellers": dev.seller_id.unique().tolist()}]
    specs += read(ROOT / "data_processed/sampling_manifest.json")[:60]
    params = read(config_path("deep_lasso"))["params"] | {"regularization": 0.0}
    for i, spec in enumerate(specs):
        if i % shards != shard:
            continue
        path = ROOT / "results/ablation_no_gradient_penalty" / f"{spec['run']}.json"
        if path.exists():
            continue
        sample = dev.loc[dev.seller_id.isin(spec["sellers"])]
        x, y, xv, yv, s, _, _ = transformed(sample, "deep_lasso", ("train", "tune"), "rank")
        start = time.monotonic()
        fitted = run_selector("deep_lasso", x, y, xv, yv, s, params, SEEDS)
        save_json(path, {"status": "ok", "case": spec["run"], "order": fitted.order,
                         "params": params, "metadata": fitted.metadata,
                         "seconds": time.monotonic() - start,
                         "fit_row_hash": digest_array(sample.loc[
                             sample.split.isin(["train", "tune"]), "row_id"].to_numpy()),
                         "post_protocol": True, "test_read": False})
        print(spec["run"], "ok", flush=True)


def inspect_vtfs():
    rows = []
    for path in sorted((ROOT / "results/runs/vtfs").glob("*.json")):
        record = read(path)
        corpus = record["metadata"]["corpus"]
        utilities = {tuple(sorted(c["features"])): c["negative_validation_mse"] for c in corpus}
        fallback = []
        for c in sorted(corpus[:96], key=lambda c: -c["negative_validation_mse"])[:8]:
            order = c["features"] + [j for j in range(31) if j not in c["features"]]
            fallback.append(order)
        winner = max(fallback, key=lambda order: utilities[tuple(sorted(order[:12]))])
        observed = utilities[tuple(sorted(record["order"][:12]))]
        no_decoder = utilities[tuple(sorted(winner[:12]))]
        rows.append({"case": record["case"], "queries": len(corpus),
                     "same_top12_as_corpus_fallback": set(winner[:12]) == set(record["order"][:12]),
                     "validation_utility_gain_from_decoded_candidates": observed - no_decoder,
                     "fallback_order": json.dumps(winner)})
    pd.DataFrame(rows).to_csv(ROOT / "results/vtfs_decoder_ablation.csv", index=False)


def summarize_ablation():
    directory = ROOT / "results/ablation_no_gradient_penalty"
    assert len(list(directory.glob("*.json"))) == 61
    full = read(directory / "reference.json")
    main = read(ROOT / "results/runs/deep_lasso/reference.json")
    rows = []
    for path in sorted(directory.glob("half*.json")):
        current = read(path)
        official = read(ROOT / "results/runs/deep_lasso" / path.name)
        assert current["fit_row_hash"] == official["fit_row_hash"]
        rows.append({"case": current["case"], "group": int(current["case"].split("_")[1]),
                     "without_penalty": jaccard(current["order"][:12], full["order"][:12]),
                     "with_penalty": jaccard(official["order"][:12], main["order"][:12])})
    table = pd.DataFrame(rows)
    groups = table.groupby("group")[["without_penalty", "with_penalty"]].mean()
    save_json(ROOT / "results/deep_lasso_ablation.json", {
        "without_penalty": cinterval(groups.without_penalty),
        "with_penalty": cinterval(groups.with_penalty),
        "paired_with_minus_without": cinterval(groups.with_penalty - groups.without_penalty),
        "full_top12_jaccard": jaccard(full["order"][:12], main["order"][:12]),
        "post_protocol": True, "hyperparameters_not_retuned": True,
        "interpretation": "local regularization ablation at one frozen configuration; not an optimized MLP comparison"})
    table.to_csv(ROOT / "results/deep_lasso_ablation_runs.csv", index=False)


def full_audit():
    manifest = read(ROOT / "data_processed/sampling_manifest.json")
    frame = load_data()
    for method in COMPARATORS:
        paths = list((ROOT / "results/runs" / method).glob("*.json"))
        assert len(paths) == 111, (method, len(paths))
        for path in paths:
            r = read(path)
            assert r["status"] == "ok" and r["test_rows_read_by_selector"] == 0
            assert sorted(r["order"]) == list(range(31))
        for spec in manifest:
            r = read(ROOT / "results/runs" / method / (spec["run"] + ".json"))
            subset = frame.loc[frame.seller_id.isin(spec["sellers"])]
            assert r["fit_row_hash"] == digest_array(subset.loc[
                subset.split.isin(["train", "tune"]), "row_id"].to_numpy())
            assert r["rank_row_hash"] == digest_array(subset.loc[
                subset.split.eq("rank"), "row_id"].to_numpy())
    simulation_count = 0
    for path in sorted((ROOT / "data_processed/simulations").glob("*.npz")):
        data = dict(np.load(path))
        for method in COMPARATORS:
            r = read(ROOT / "results/simulation_runs" / method / (path.stem + ".json"))
            assert r["status"] == "ok" and r["data_hash"] == digest_array(data["x"])
            assert np.array_equal(r["truth"], data["truth"])
            assert sorted(r["order"]) == list(range(31))
            if "response_hash" in r:
                assert r["response_hash"] == digest_array(data["y"])
            if method in ("elastic_net", "stability_selection"):
                assert r["response_scaling"] == "training mean and population standard deviation"
            truth = set(map(int, data["truth"]))
            for m in r["metrics"]:
                if m["rule"].startswith("top"):
                    k = int(m["rule"][3:])
                    found = set(r["order"][:k])
                    assert m["discoveries"] == k
                    assert np.isclose(m["FDP"], len(found - truth) / k)
                    assert np.isclose(m["power"], len(found & truth) / len(truth))
                assert 0 <= m["FDP"] <= 1 and 0 <= m["power"] <= 1
            simulation_count += 1
    assert simulation_count == 4800
    predictions = list((ROOT / "results/predictions").glob("*.csv"))
    assert len(predictions) == 82
    for path in predictions:
        d = pd.read_csv(path)
        meta = read(path.with_suffix(".json"))
        assert not meta["test_used_for_training"]
        assert len(d) == 3413
        assert np.array_equal(d.row_id, frame.loc[frame.split.eq("test"), "row_id"])
        assert np.isfinite(d.prediction).all()
    save_json(ROOT / "results/final_audit.json", {
        "passed": True, "real_comparator_runs": 888, "simulation_comparator_runs": simulation_count,
        "prediction_outputs": len(predictions), "common_fit_and_rank_row_hashes": "passed",
        "common_simulation_x_truth_and_available_y_hashes": "passed",
        "fixed_budget_fdp_power_recomputed": "passed",
        "source_code_sha256": {p.name: digest_file(p) for p in sorted((ROOT / "code").glob("*.py"))},
        "protocol_sha256": digest_file(ROOT / "protocol.json"),
        "panel_sha256": digest_file(ROOT / "data_processed/seller_month_asof.parquet"),
        "sampling_manifest_sha256": digest_file(ROOT / "data_processed/sampling_manifest.json"),
        "note": "Diagnostic failures are not counted as successful full comparisons; run counts do not establish statistical validity."})
    print("Final input and output audit passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ablation", "summarize", "audit"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "ablation":
        deep_lasso_ablation(args.shard, args.shards)
    elif args.stage == "summarize":
        inspect_vtfs()
        summarize_ablation()
    else:
        full_audit()
