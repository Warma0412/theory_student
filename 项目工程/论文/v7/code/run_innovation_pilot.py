"""Exploratory environment-consistent Deep Lasso and feature-utility audit.

This is a post-V7 pilot. It never overwrites the frozen V7 comparison and must
not be described as preregistered confirmation or as globally novel.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch import nn

from common import (
    ROOT, FEATURES, SEEDS, Preprocessor, digest_array, jaccard, load_data,
    rank, save_json, seed_all,
)
from models import fit_neural, neural_predict, xgb_model
from run_experiments import simulation_data
from validate_and_summarize import cinterval, nogueira, read

PILOT = ROOT / "innovation_pilot"
RESULTS = PILOT / "results"
RUNS = RESULTS / "runs"
SIMRUNS = RESULTS / "simulations"
LABELS = dict(zip(
    FEATURES,
    pd.read_csv(ROOT / "results/feature_dictionary.csv").label_zh,
))
K = 12
GAMMAS = [0.0, 0.1, 1.0, 10.0]
SCENARIOS = ["linear_weak", "nonlinear", "nonlinear_panel"]
REPETITIONS = 30


def make_model(p=31, width=64, layers=2):
    parts = []
    current = p
    for _ in range(layers):
        parts += [nn.Linear(current, width), nn.ReLU(), nn.Dropout(.1)]
        current = width
    return nn.Sequential(*parts, nn.Linear(current, 1))


def predict(model, x):
    model.eval()
    with torch.no_grad():
        out = torch.cat([
            model(batch) for batch in torch.as_tensor(
                x, dtype=torch.float32
            ).split(1024)
        ]).squeeze(-1)
    return out.numpy() * model.y_std + model.y_mean


def importance(model, x, y, environments=None):
    """Average L1-normalized input-loss-gradient norms across environments."""
    target = (np.asarray(y) - model.y_mean) / model.y_std
    if environments is None:
        environments = np.zeros(len(x), dtype=int)
    rows = []
    for environment in np.unique(environments):
        idx = np.flatnonzero(environments == environment)
        grads = []
        for begin in range(0, len(idx), 1024):
            take = idx[begin:begin + 1024]
            xb = torch.as_tensor(x[take], dtype=torch.float32).clone().requires_grad_(True)
            yb = torch.as_tensor(target[take], dtype=torch.float32)
            loss = (model(xb).squeeze(-1) - yb).square().mean()
            grad, = torch.autograd.grad(loss, xb)
            grads.append(grad.detach())
        score = torch.cat(grads).square().sum(0).sqrt().numpy()
        rows.append(score / max(score.sum(), 1e-12))
    return np.mean(rows, axis=0)


def fit_consistent(
    x, y, months, sellers, xv, yv, gamma, seed,
    regularization=1.0, epochs=100, patience_limit=12,
):
    """Train with complementary-seller gradient-importance consistency."""
    seed_all(seed)
    model = make_model(x.shape[1])
    xt = torch.as_tensor(x, dtype=torch.float32)
    mean, std = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    yt = torch.as_tensor((y - mean) / std, dtype=torch.float32)
    xvt = torch.as_tensor(xv, dtype=torch.float32)
    yvt = torch.as_tensor((yv - mean) / std, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=2e-4)
    unique_sellers = np.unique(sellers)
    unique_months = np.unique(months)
    best, stale, state, history = np.inf, 0, None, []
    started = time.monotonic()
    steps = max(8, math.ceil(len(x) / 512))
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + 1009 * epoch)
        shuffled = rng.permutation(unique_sellers)
        left_sellers = shuffled[:len(shuffled) // 2]
        left = np.isin(sellers, left_sellers)
        eligible = [
            month for month in unique_months
            if np.sum((months == month) & left) >= 16
            and np.sum((months == month) & ~left) >= 16
        ]
        if not eligible:
            raise ValueError("No month has enough rows in both seller halves")
        model.train()
        epoch_rows = []
        for step in range(steps):
            month = eligible[(epoch * steps + step) % len(eligible)]
            ia = np.flatnonzero((months == month) & left)
            ib = np.flatnonzero((months == month) & ~left)
            ia = rng.choice(ia, min(256, len(ia)), replace=False)
            ib = rng.choice(ib, min(256, len(ib)), replace=False)
            xa = xt[ia].clone().requires_grad_(True)
            xb = xt[ib].clone().requires_grad_(True)
            loss_a = (model(xa).squeeze(-1) - yt[ia]).square().mean()
            loss_b = (model(xb).squeeze(-1) - yt[ib]).square().mean()
            grad_a, = torch.autograd.grad(loss_a, xa, create_graph=True)
            grad_b, = torch.autograd.grad(loss_b, xb, create_graph=True)
            score_a = grad_a.square().sum(0).add(1e-8).sqrt()
            score_b = grad_b.square().sum(0).add(1e-8).sqrt()
            sparse = .5 * (score_a.sum() + score_b.sum())
            probability_a = score_a / score_a.sum()
            probability_b = score_b / score_b.sum()
            consistency = torch.abs(probability_a - probability_b).sum()
            prediction_loss = .5 * (loss_a + loss_b)
            total = prediction_loss + regularization * sparse + gamma * consistency
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_rows.append([
                float(prediction_loss.detach()),
                float(sparse.detach()),
                float(consistency.detach()),
                float(total.detach()),
            ])
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(batch) for batch in xvt.split(1024)]).squeeze(-1)
            valid = float((pv - yvt).square().mean())
        history.append([epoch + 1, *np.mean(epoch_rows, axis=0).tolist(), valid])
        if valid < best - 1e-6:
            best, stale = valid, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience_limit:
            break
    model.load_state_dict(state)
    model.eval()
    model.y_mean, model.y_std = mean, std
    return model, {
        "seed": seed, "gamma": gamma, "regularization": regularization,
        "epochs": len(history), "best_validation_mse_scaled": best,
        "history": history, "seconds": time.monotonic() - started,
        "parameters": sum(v.numel() for v in model.parameters()),
        "objective": "prediction + Deep-Lasso gradient sparsity + complementary-seller gradient-distribution L1",
    }


def prepare(frame, fit_splits, valid_split):
    tr = frame.loc[frame.split.isin(fit_splits)].copy()
    va = frame.loc[frame.split.eq(valid_split)].copy()
    pre = Preprocessor().fit(tr[FEATURES].to_numpy(float))
    return (
        pre.transform(tr[FEATURES].to_numpy(float)),
        tr.log_gmv_next_month.to_numpy(float),
        tr.target_month.astype(str).to_numpy(),
        tr.seller_id.to_numpy(),
        pre.transform(va[FEATURES].to_numpy(float)),
        va.log_gmv_next_month.to_numpy(float),
        va.target_month.astype(str).to_numpy(),
        va.seller_id.to_numpy(),
        tr, va, pre,
    )


def seller_partition_jaccard(models, x, y, sellers, repeats=10):
    values = []
    unique = np.unique(sellers)
    for repeat in range(repeats):
        order = np.random.default_rng(7700 + repeat).permutation(unique)
        halves = np.array_split(order, 2)
        selected = []
        for half in halves:
            idx = np.isin(sellers, half)
            scores = np.mean([
                importance(model, x[idx], y[idx]) for model in models
            ], axis=0)
            selected.append(rank(scores)[:K])
        values.append(jaccard(*selected))
    return values


def tune():
    destination = RESULTS / "tuning.json"
    if destination.exists():
        return read(destination)
    frame = load_data()
    x, y, months, sellers, xv, yv, vm, vs, *_ = prepare(
        frame, ("train",), "tune"
    )
    trials = []
    for gamma in GAMMAS:
        models, fits, predictions = [], [], []
        for seed in SEEDS:
            model, meta = fit_consistent(
                x, y, months, sellers, xv, yv, gamma, seed
            )
            models.append(model)
            fits.append(meta)
            predictions.append(predict(model, xv))
        rmse = float(mean_squared_error(yv, np.mean(predictions, axis=0)) ** .5)
        partitions = seller_partition_jaccard(models, xv, yv, vs)
        month_orders = []
        for month in np.unique(vm):
            idx = vm == month
            score = np.mean([
                importance(model, xv[idx], yv[idx]) for model in models
            ], axis=0)
            month_orders.append(rank(score)[:K])
        trials.append({
            "gamma": gamma, "tune_rmse_log": rmse,
            "tune_seller_half_jaccard": float(np.mean(partitions)),
            "tune_seller_half_jaccard_values": partitions,
            "tune_month_jaccard": jaccard(*month_orders),
            "fits": fits,
        })
        save_json(destination, {"status": "running", "trials": trials})
        print("gamma", gamma, "rmse", rmse, "seller J", np.mean(partitions),
              "month J", trials[-1]["tune_month_jaccard"], flush=True)
    best_rmse = min(row["tune_rmse_log"] for row in trials)
    eligible = [
        row for row in trials if row["tune_rmse_log"] <= best_rmse * 1.01
    ]
    chosen = sorted(
        eligible,
        key=lambda row: (
            -row["tune_seller_half_jaccard"],
            row["tune_rmse_log"],
            row["gamma"],
        ),
    )[0]
    output = {
        "status": "ok", "trials": trials, "chosen_gamma": chosen["gamma"],
        "rule": "highest tune complementary-seller top12 Jaccard among gamma values within 1% of best tune RMSE; ties by RMSE then smaller gamma",
        "post_v7_exploratory": True, "test_read_for_choice": False,
    }
    save_json(destination, output)
    return output


def execute_case(case_name, frame, seeds=SEEDS):
    destination = RUNS / f"{case_name}.json"
    if destination.exists():
        return read(destination)
    gamma = tune()["chosen_gamma"]
    x, y, months, sellers, xv, yv, vm, vs, tr, va, _ = prepare(
        frame, ("train", "tune"), "rank"
    )
    models, fits, scores = [], [], []
    for seed in seeds:
        model, meta = fit_consistent(
            x, y, months, sellers, xv, yv, gamma, seed
        )
        models.append(model)
        fits.append(meta)
        scores.append(importance(model, xv, yv, vm))
    score = np.mean(scores, axis=0)
    output = {
        "status": "ok", "case": case_name, "gamma": gamma,
        "score": score, "order": rank(score), "fits": fits,
        "fit_row_hash": digest_array(tr.row_id.to_numpy()),
        "rank_row_hash": digest_array(va.row_id.to_numpy()),
        "ranking_months": sorted(np.unique(vm).tolist()),
        "post_v7_exploratory": True, "test_rows_read": 0,
    }
    save_json(destination, output)
    return output


def run_stability(shard=0, shards=1):
    frame = load_data()
    dev = frame.loc[frame.split.ne("test")].copy()
    execute_case("reference", dev)
    manifest = read(ROOT / "data_processed/sampling_manifest.json")[:60]
    for index, spec in enumerate(manifest):
        if index % shards != shard:
            continue
        execute_case(
            spec["run"],
            dev.loc[dev.seller_id.isin(spec["sellers"])].copy(),
        )
        print(spec["run"], "ok", flush=True)


def summarize_stability():
    manifest = read(ROOT / "data_processed/sampling_manifest.json")[:60]
    reference = read(RUNS / "reference.json")
    rows, pair_rows = [], []
    sets = []
    for spec in manifest:
        row = read(RUNS / f"{spec['run']}.json")
        current = row["order"][:K]
        sets.append(current)
        rows.append({
            "run": spec["run"], "group": spec["group"], "half": spec["half"],
            "reference_jaccard": jaccard(current, reference["order"][:K]),
            "features": json.dumps(current),
        })
    detail = pd.DataFrame(rows)
    for group, part in detail.groupby("group"):
        selected = [
            json.loads(v) for v in part.sort_values("half").features
        ]
        pair_rows.append({
            "group": group,
            "complementary_jaccard": jaccard(*selected),
        })
    grouped = detail.groupby("group").reference_jaccard.mean()
    summary = {
        "reference_jaccard": cinterval(grouped),
        "complementary_jaccard": cinterval(
            [row["complementary_jaccard"] for row in pair_rows]
        ),
        "nogueira": nogueira(sets),
        "completed_half_samples": len(rows),
        "k": K,
    }
    detail.to_csv(RESULTS / "stability_runs.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(
        RESULTS / "complementary_pair_runs.csv", index=False
    )
    save_json(RESULTS / "stability_summary.json", summary)
    return summary


def evaluate_prediction():
    destination = RESULTS / "prediction.json"
    if destination.exists():
        return read(destination)
    order = read(RUNS / "reference.json")["order"]
    frame = load_data()
    train = frame.loc[frame.split.isin(["train", "tune"])]
    valid = frame.loc[frame.split.eq("rank")]
    test = frame.loc[frame.split.eq("test")]
    pre = Preprocessor().fit(train[FEATURES].to_numpy(float))
    columns = order[:K]
    x = pre.transform(train[FEATURES].to_numpy(float))[:, columns]
    xv = pre.transform(valid[FEATURES].to_numpy(float))[:, columns]
    xt = pre.transform(test[FEATURES].to_numpy(float))[:, columns]
    y = train.log_gmv_next_month.to_numpy(float)
    yv = valid.log_gmv_next_month.to_numpy(float)
    yt = test.log_gmv_next_month.to_numpy(float)
    output = {"k": K, "features": [FEATURES[i] for i in columns]}
    for evaluator, config_name in (
        ("xgboost", "xgboost_shap"), ("tabm", "tabm")
    ):
        config = read(ROOT / f"results/configs/{config_name}.json")["params"]
        valid_predictions, test_predictions = [], []
        for seed in SEEDS:
            if evaluator == "xgboost":
                model = xgb_model(config, seed).fit(x, y)
                valid_predictions.append(model.predict(xv))
                model = xgb_model(config, seed).fit(
                    np.concatenate([x, xv]), np.r_[y, yv]
                )
                test_predictions.append(model.predict(xt))
            else:
                model, meta = fit_neural(
                    x, y, xv, yv, config, "tabm", seed
                )
                valid_predictions.append(neural_predict(model, xv))
                best_epoch = min(meta["history"], key=lambda row: row[2])[0]
                model, _ = fit_neural(
                    np.concatenate([x, xv]), np.r_[y, yv], xv, yv,
                    config | {"epochs": best_epoch, "fixed_epochs": True},
                    "tabm", seed,
                )
                test_predictions.append(neural_predict(model, xt))
        output[evaluator] = {
            "validation_rmse": float(mean_squared_error(
                yv, np.mean(valid_predictions, axis=0)
            ) ** .5),
            "test_rmse": float(mean_squared_error(
                yt, np.mean(test_predictions, axis=0)
            ) ** .5),
        }
    output.update({
        "post_v7_exploratory": True,
        "test_not_used_for_tuning": True,
        "existing_test_previously_observed": True,
    })
    save_json(destination, output)
    return output


def run_simulations(shard=0, shards=1):
    protocol_scenarios = {
        row["name"]: row
        for row in read(ROOT / "protocol.json")["simulation"]["scenarios"]
    }
    gamma = tune()["chosen_gamma"]
    for scenario_name in SCENARIOS:
        scenario = protocol_scenarios[scenario_name]
        for repetition in range(REPETITIONS):
            index = SCENARIOS.index(scenario_name) * REPETITIONS + repetition
            if index % shards != shard:
                continue
            destination = SIMRUNS / f"{scenario_name}_{repetition:03d}.json"
            if destination.exists():
                continue
            data = simulation_data(scenario, repetition)
            x, y = data["x"], data["y"]
            pre = Preprocessor(business=False).fit(x[:1200])
            xt = pre.transform(x[:1200])
            xv = pre.transform(x[1200:1600])
            pseudo_months = np.zeros(1200, dtype=int)
            pseudo_sellers = data["groups"][:1200]
            if len(np.unique(pseudo_sellers)) < 4:
                pseudo_sellers = np.arange(1200)
            scores, fits = [], []
            for seed in SEEDS:
                model, meta = fit_consistent(
                    xt, y[:1200], pseudo_months, pseudo_sellers,
                    xv, y[1200:1600], gamma, seed,
                )
                scores.append(importance(model, xv, y[1200:1600]))
                fits.append(meta)
            score = np.mean(scores, axis=0)
            selected = set(map(int, rank(score)[:K]))
            truth = set(map(int, data["truth"]))
            fdp = len(selected - truth) / K
            power = len(selected & truth) / len(truth)
            save_json(destination, {
                "status": "ok", "scenario": scenario_name,
                "repetition": repetition, "gamma": gamma,
                "order": rank(score), "FDP": fdp, "power": power,
                "data_hash": digest_array(x), "truth": data["truth"],
                "fits": fits, "post_v7_exploratory": True,
            })
            print(scenario_name, repetition, "ok", flush=True)


def summarize_simulations():
    rows = []
    for scenario in SCENARIOS:
        for repetition in range(REPETITIONS):
            row = read(SIMRUNS / f"{scenario}_{repetition:03d}.json")
            rows.append({
                "scenario": scenario, "repetition": repetition,
                "FDP": row["FDP"], "power": row["power"],
            })
    detail = pd.DataFrame(rows)
    output = []
    old = pd.read_csv(ROOT / "results/simulation_summary.csv")
    for scenario, part in detail.groupby("scenario"):
        current = {"method": "EC-DeepLasso", "scenario": scenario}
        for metric in ("FDP", "power"):
            current.update({
                metric + "_" + key: value
                for key, value in cinterval(part[metric]).items()
                if key != "n"
            })
        output.append(current)
        for method in ("deep_lasso", "xgboost_shap", "tabm"):
            base = old.loc[
                old.method.eq(method)
                & old.scenario.eq(scenario)
                & old.rule.eq("top12")
            ].iloc[0]
            output.append({
                "method": method, "scenario": scenario,
                "FDP_mean": base.FDP_mean,
                "power_mean": base.power_mean,
                "note": "V7 100-repetition reference; comparison is descriptive because pilot uses first 30 repetitions",
            })
    detail.to_csv(RESULTS / "simulation_runs.csv", index=False)
    pd.DataFrame(output).to_csv(
        RESULTS / "simulation_summary.csv", index=False
    )


def rolling_feature_audit():
    destination = RESULTS / "feature_utility_audit.csv"
    if destination.exists():
        return pd.read_csv(destination)
    frame = load_data().loc[lambda d: d.split.ne("test")].copy()
    selected = list(map(int, read(RUNS / "reference.json")["order"][:K]))
    unselected = [j for j in range(len(FEATURES)) if j not in selected]
    months = sorted(frame.target_month.unique())
    validation_months = months[7:]
    folds = []
    config = read(ROOT / "results/configs/xgboost_shap.json")["params"]
    for month in validation_months:
        train = frame.loc[frame.target_month < month]
        valid = frame.loc[frame.target_month == month]
        pre = Preprocessor().fit(train[FEATURES].to_numpy(float))
        folds.append({
            "month": str(month),
            "x": pre.transform(train[FEATURES].to_numpy(float)),
            "y": train.log_gmv_next_month.to_numpy(float),
            "xv": pre.transform(valid[FEATURES].to_numpy(float)),
            "yv": valid.log_gmv_next_month.to_numpy(float),
        })
    cache = {}

    def losses(columns):
        key = tuple(sorted(map(int, columns)))
        if key not in cache:
            values = []
            for fold_index, fold in enumerate(folds):
                predictions = []
                for seed in SEEDS:
                    model = xgb_model(config, seed).fit(
                        fold["x"][:, key], fold["y"]
                    )
                    predictions.append(model.predict(fold["xv"][:, key]))
                values.append(mean_squared_error(
                    fold["yv"], np.mean(predictions, axis=0)
                ) ** .5)
            cache[key] = np.array(values)
        return cache[key]

    baseline = losses(selected)
    stability = pd.read_csv(RESULTS / "stability_runs.csv")
    frequency = np.zeros(len(FEATURES))
    for encoded in stability.features:
        frequency[json.loads(encoded)] += 1
    frequency /= len(stability)
    strong_methods = ("xgboost_shap", "deep_lasso", "tabm", "shadow_trees")
    consensus = np.zeros(len(FEATURES), dtype=int)
    for method in strong_methods:
        order = read(ROOT / f"results/runs/{method}/reference.json")["order"][:K]
        consensus[order] += 1
    development = frame.loc[frame.split.isin(["train", "tune", "rank"])]
    correlation = development[FEATURES].corr(method="spearman").abs()
    dictionary = pd.read_csv(ROOT / "results/feature_dictionary.csv").set_index(
        "feature"
    )
    rows = []
    for feature in selected:
        reduced = [j for j in selected if j != feature]
        drop_loss = losses(reduced)
        candidate_losses = {
            replacement: losses(reduced + [replacement])
            for replacement in unselected
        }
        nested_losses, nested_replacements = [], []
        for fold_index in range(2, len(folds)):
            replacement = min(
                unselected,
                key=lambda item: candidate_losses[item][:fold_index].mean(),
            )
            nested_replacements.append(replacement)
            nested_losses.append(candidate_losses[replacement][fold_index])
        others = [FEATURES[j] for j in selected if j != feature]
        feature_name = FEATURES[feature]
        most_correlated = correlation.loc[feature_name, others].idxmax()
        rows.append({
            "feature_index": feature, "feature": feature_name,
            "label_zh": LABELS[feature_name],
            "half_sample_frequency": frequency[feature],
            "strong_method_consensus_0_to_4": consensus[feature],
            "drop_delta_rmse_mean": float(np.mean(drop_loss - baseline)),
            "drop_positive_month_fraction": float(np.mean(drop_loss > baseline)),
            "nested_replacement_delta_rmse_mean": float(np.mean(
                np.asarray(nested_losses) - baseline[2:]
            )),
            "nested_replacements": "|".join(
                FEATURES[j] for j in nested_replacements
            ),
            "max_abs_spearman_with_selected": float(
                correlation.loc[feature_name, most_correlated]
            ),
            "most_correlated_selected": most_correlated,
            "missing_fraction": float(
                dictionary.loc[feature_name, "missing_fraction"]
            ),
            "audit_months": "|".join(fold["month"] for fold in folds),
        })
    result = pd.DataFrame(rows).sort_values(
        ["nested_replacement_delta_rmse_mean", "half_sample_frequency"],
        ascending=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    save_json(RESULTS / "feature_utility_audit_metadata.json", {
        "base_features": [FEATURES[j] for j in selected],
        "rolling_validation_months": [fold["month"] for fold in folds],
        "replacement_rule": "for each audit month after the first two folds, choose the best replacement using only preceding rolling folds",
        "drop_delta_positive_means_feature_removal_worsened_rmse": True,
        "replacement_delta_positive_means_best_historical_replacement_worsened_rmse": True,
        "evaluator": "same frozen XGBoost configuration, three seeds",
        "post_v7_exploratory": True,
    })
    return result


def build_report():
    tuning = read(RESULTS / "tuning.json")
    stability = read(RESULTS / "stability_summary.json")
    prediction = read(RESULTS / "prediction.json")
    audit = pd.read_csv(RESULTS / "feature_utility_audit.csv")
    simulation = pd.read_csv(RESULTS / "simulation_summary.csv")
    baseline_stability = pd.read_csv(
        ROOT / "results/stability_summary.csv"
    )
    baseline_stability = baseline_stability.loc[
        baseline_stability.k.eq(K)
        & baseline_stability.fraction.eq(.5)
        & baseline_stability.method.isin(
            ["deep_lasso", "xgboost_shap", "shadow_trees"]
        ),
        ["method", "mean", "nogueira"],
    ]
    report = [
        "# V7后验创新探索：时间环境一致性Deep Lasso与变量效用审计",
        "",
        "本试验发生在V7测试结果已经查看之后，只能作为探索性证据。它不修改V7主结果，也不能宣称预先注册成功。",
        "",
        "## 方法定位",
        "",
        "在Deep Lasso的预测损失与输入梯度稀疏惩罚之外，增加同一月份两个互补卖家子群的归一化输入梯度分布L1距离。该项直接惩罚“模型在不同卖家群上依赖不同指标”。",
        "",
        "这一构造受到Deep Lasso、稳定特征选择及跨子集一致性正则研究启发。当前检索未发现与本实现完全相同的卖家互补、同月、输入损失梯度分布约束，但不能据有限检索宣称世界首创。",
        "",
        "## 开发期调参",
        "",
        pd.DataFrame(tuning["trials"])[[
            "gamma", "tune_rmse_log", "tune_seller_half_jaccard",
            "tune_month_jaccard"
        ]].to_markdown(index=False),
        "",
        f"冻结规则选择 gamma={tuning['chosen_gamma']}。测试目标没有参与gamma选择。",
        "",
        "## 半样本稳定性",
        "",
        pd.concat([
            pd.DataFrame([{
                "method": "EC-DeepLasso",
                "mean": stability["reference_jaccard"]["mean"],
                "nogueira": stability["nogueira"],
            }]),
            baseline_stability,
        ]).to_markdown(index=False),
        "",
        "## 预测结果（后验描述）",
        "",
        pd.DataFrame([
            {"evaluator": key, **value}
            for key, value in prediction.items()
            if key in ("xgboost", "tabm")
        ]).to_markdown(index=False),
        "",
        "现有2018年测试期此前已经被查看，因此这里不能作为新方法的独立确认结果。",
        "",
        "## 代表性模拟",
        "",
        simulation.to_markdown(index=False),
        "",
        "## 变量效用审计",
        "",
        audit[[
            "label_zh", "half_sample_frequency",
            "strong_method_consensus_0_to_4", "drop_delta_rmse_mean",
            "drop_positive_month_fraction",
            "nested_replacement_delta_rmse_mean",
            "max_abs_spearman_with_selected", "missing_fraction",
        ]].to_markdown(index=False),
        "",
        "判断变量是否值得保留，至少应同时检查：样本频率、跨方法共识、删除损失、可替代性、时间方向一致性、冗余和数据可得性。任何单项领先都不构成自动准入。",
    ]
    (PILOT / "探索结果.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "tune", "stability", "summarize_stability", "predict",
            "simulate", "summarize_simulations", "audit", "report", "all",
        ],
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    PILOT.mkdir(exist_ok=True)
    if args.stage in ("tune", "all"):
        tune()
    if args.stage in ("stability", "all"):
        run_stability(args.shard, args.shards)
    if args.stage in ("summarize_stability", "all"):
        summarize_stability()
    if args.stage in ("predict", "all"):
        evaluate_prediction()
    if args.stage in ("simulate", "all"):
        run_simulations(args.shard, args.shards)
    if args.stage in ("summarize_simulations", "all"):
        summarize_simulations()
    if args.stage in ("audit", "all"):
        rolling_feature_audit()
    if args.stage in ("report", "all"):
        build_report()


if __name__ == "__main__":
    main()
