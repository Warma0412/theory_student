"""Model-aware but selector-balanced audit of whether selected variables are useful.

The audit uses development months only. It evaluates all 31 variables and all
features selected by four strong but structurally different selectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from common import ROOT, FEATURES, SEEDS, Preprocessor, load_data, save_json
from models import fit_neural, neural_predict, xgb_model
from validate_and_summarize import LABELS, read

PILOT = ROOT / "innovation_pilot"
RESULTS = PILOT / "results"
METHODS = ["xgboost_shap", "deep_lasso", "tabm", "shadow_trees"]
K = 12
AUDIT_SEEDS = [11]


def rolling_folds():
    frame = load_data().loc[lambda d: d.split.ne("test")].copy()
    months = sorted(frame.target_month.unique())
    folds = []
    for month in months[7:]:
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
    return frame, folds


def run():
    output = RESULTS / "variable_quality_by_selector.csv"
    aggregate_output = RESULTS / "variable_quality_summary.csv"
    if output.exists() and aggregate_output.exists():
        return
    frame, folds = rolling_folds()
    config = read(ROOT / "results/configs/xgboost_shap.json")["params"]
    cache = {}

    def losses(columns):
        key = tuple(sorted(map(int, columns)))
        if key not in cache:
            values = []
            for fold in folds:
                predictions = []
                for seed in AUDIT_SEEDS:
                    model = xgb_model(config, seed).fit(
                        fold["x"][:, key], fold["y"]
                    )
                    predictions.append(model.predict(fold["xv"][:, key]))
                values.append(mean_squared_error(
                    fold["yv"], np.mean(predictions, axis=0)
                ) ** .5)
            cache[key] = np.asarray(values)
        return cache[key]

    all_features = list(range(len(FEATURES)))
    all_loss = losses(all_features)
    full_rows = []
    for feature in all_features:
        reduced = [j for j in all_features if j != feature]
        delta = losses(reduced) - all_loss
        full_rows.append({
            "feature_index": feature,
            "feature": FEATURES[feature],
            "full_model_drop_delta_mean": float(delta.mean()),
            "full_model_drop_positive_month_fraction": float(
                np.mean(delta > 0)
            ),
            "full_model_drop_delta_min": float(delta.min()),
            "full_model_drop_delta_max": float(delta.max()),
        })
        print("full drop", feature, flush=True)

    selector_rows = []
    for method in METHODS:
        selected = list(map(
            int,
            read(ROOT / f"results/runs/{method}/reference.json")["order"][:K],
        ))
        unselected = [j for j in all_features if j not in selected]
        baseline = losses(selected)
        for feature in selected:
            reduced = [j for j in selected if j != feature]
            drop_delta = losses(reduced) - baseline
            candidate = {
                replacement: losses(reduced + [replacement])
                for replacement in unselected
            }
            nested_delta, nested_choice = [], []
            # Each audit month chooses its replacement using preceding months.
            for fold_index in range(2, len(folds)):
                replacement = min(
                    unselected,
                    key=lambda item: candidate[item][:fold_index].mean(),
                )
                nested_choice.append(replacement)
                nested_delta.append(
                    candidate[replacement][fold_index] - baseline[fold_index]
                )
            selector_rows.append({
                "selector": method,
                "feature_index": feature,
                "feature": FEATURES[feature],
                "drop_delta_mean": float(drop_delta.mean()),
                "drop_positive_month_fraction": float(
                    np.mean(drop_delta > 0)
                ),
                "nested_replacement_delta_mean": float(
                    np.mean(nested_delta)
                ),
                "nested_replacement_positive_fraction": float(
                    np.mean(np.asarray(nested_delta) > 0)
                ),
                "nested_replacements": "|".join(
                    FEATURES[j] for j in nested_choice
                ),
                "baseline_rmse_mean": float(baseline.mean()),
            })
            print(method, feature, flush=True)
    selector_table = pd.DataFrame(selector_rows)
    selector_table.to_csv(output, index=False)

    stability = pd.read_csv(ROOT / "results/stability_runs.csv")
    stability = stability.loc[
        stability.k.eq(K)
        & stability.fraction.eq(.5)
        & stability.method.isin(METHODS)
    ]
    frequency = {}
    for method in METHODS:
        values = np.zeros(len(FEATURES))
        part = stability.loc[stability.method.eq(method)]
        for encoded in part.features:
            values[json.loads(encoded)] += 1
        frequency[method] = values / len(part)

    dictionary = pd.read_csv(
        ROOT / "results/feature_dictionary.csv"
    ).set_index("feature")
    correlation = frame[FEATURES].corr(method="spearman").abs()
    selection_sets = {
        method: set(map(
            int,
            read(ROOT / f"results/runs/{method}/reference.json")["order"][:K],
        ))
        for method in METHODS
    }
    full_table = pd.DataFrame(full_rows).set_index("feature_index")
    aggregate = []
    for feature in all_features:
        name = FEATURES[feature]
        selected_rows = selector_table.loc[
            selector_table.feature_index.eq(feature)
        ]
        correlations = correlation.loc[
            name, [column for column in FEATURES if column != name]
        ]
        method_frequency = [
            frequency[method][feature] for method in METHODS
        ]
        method_count = sum(
            feature in selection_sets[method] for method in METHODS
        )
        replacement_delta = (
            float(selected_rows.nested_replacement_delta_mean.mean())
            if len(selected_rows) else np.nan
        )
        drop_delta = float(full_table.loc[
            feature, "full_model_drop_delta_mean"
        ])
        drop_fraction = float(full_table.loc[
            feature, "full_model_drop_positive_month_fraction"
        ])
        mean_frequency = float(np.mean(method_frequency))
        if (
            method_count >= 3
            and mean_frequency >= .70
            and drop_fraction >= .60
            and np.isfinite(replacement_delta)
            and replacement_delta > 0
        ):
            category = "核心候选"
        elif method_count >= 2 and (
            correlations.max() >= .80
            or (
                np.isfinite(replacement_delta)
                and replacement_delta <= 0
            )
        ):
            category = "可替代候选"
        elif method_count >= 2 or drop_fraction >= .60:
            category = "观察候选"
        else:
            category = "证据不足"
        aggregate.append({
            "feature_index": feature, "feature": name,
            "label_zh": dictionary.loc[name, "label_zh"],
            "quality_category": category,
            "selected_by_strong_methods_0_to_4": method_count,
            "mean_half_sample_frequency": mean_frequency,
            "min_half_sample_frequency": float(np.min(method_frequency)),
            "max_half_sample_frequency": float(np.max(method_frequency)),
            "full_model_drop_delta_mean": drop_delta,
            "full_model_drop_positive_month_fraction": drop_fraction,
            "mean_nested_replacement_delta_when_selected": replacement_delta,
            "max_abs_spearman": float(correlations.max()),
            "most_correlated_feature": correlations.idxmax(),
            "missing_fraction": float(
                dictionary.loc[name, "missing_fraction"]
            ),
        })
    aggregate_table = pd.DataFrame(aggregate).sort_values(
        [
            "quality_category",
            "selected_by_strong_methods_0_to_4",
            "mean_half_sample_frequency",
        ],
        ascending=[True, False, False],
    )
    aggregate_table.to_csv(aggregate_output, index=False)
    save_json(RESULTS / "variable_quality_metadata.json", {
        "development_only": True,
        "rolling_validation_months": [fold["month"] for fold in folds],
        "selectors": METHODS,
        "common_evaluator": "frozen XGBoost; one fixed seed for the exploratory exhaustive swap audit",
        "fixed_k": K,
        "core_rule": "selected by >=3/4 strong selectors; mean half-sample frequency >=0.70; full-model removal worsens >=60% months; nested historical replacement has positive mean regret",
        "replaceable_rule": "selected by >=2/4 and max Spearman >=0.80 or nested replacement regret <=0",
        "warning": "Categories are a transparent pilot admission rule, not probabilities of truth or causal effects.",
        "post_v7_exploratory": True,
    })


def report():
    summary = pd.read_csv(RESULTS / "variable_quality_summary.csv")
    detail = pd.read_csv(RESULTS / "variable_quality_by_selector.csv")
    tuning = read(RESULTS / "tuning.json")
    pilot_prediction = read(RESULTS / "prediction.json")
    set_prediction = pd.read_csv(RESULTS / "audit_set_prediction_v2.csv")
    paired = read(RESULTS / "group_aware_top8_paired_intervals.json")
    v7_prediction = pd.read_csv(ROOT / "results/prediction_summary.csv")
    def markdown(frame):
        values = frame.fillna("").astype(str)
        header = "| " + " | ".join(values.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
        rows = [
            "| " + " | ".join(row) + " |"
            for row in values.to_numpy().tolist()
        ]
        return "\n".join([header, separator, *rows])
    sensitivity = []
    for _, row in summary.iterrows():
        selected = detail.loc[detail.feature.eq(row.feature)]
        replacement_median = (
            float(selected.nested_replacement_delta_mean.median())
            if len(selected) else np.nan
        )
        replacement_positive_fraction = (
            float(np.mean(selected.nested_replacement_delta_mean > 0))
            if len(selected) else np.nan
        )
        if (
            row.selected_by_strong_methods_0_to_4 >= 3
            and row.mean_half_sample_frequency >= .70
            and row.full_model_drop_positive_month_fraction >= .60
            and replacement_median > 0
            and row.max_abs_spearman < .80
        ):
            robust_category = "核心候选"
        elif row.selected_by_strong_methods_0_to_4 >= 2 and (
            row.max_abs_spearman >= .80
            or (
                np.isfinite(replacement_median)
                and replacement_median <= 0
            )
        ):
            robust_category = "可替代候选"
        elif (
            row.selected_by_strong_methods_0_to_4 >= 2
            or row.full_model_drop_positive_month_fraction >= .60
        ):
            robust_category = "观察候选"
        else:
            robust_category = "证据不足"
        sensitivity.append({
            "label_zh": row.label_zh,
            "strict_mean_category": row.quality_category,
            "robust_majority_category": robust_category,
            "replacement_median_across_selectors": replacement_median,
            "replacement_positive_selector_fraction": replacement_positive_fraction,
            "max_abs_spearman": row.max_abs_spearman,
        })
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(
        RESULTS / "variable_quality_rule_sensitivity.csv", index=False
    )
    lines = [
        "# 创新探索与变量质量审计：初步结论",
        "",
        "## 1. 模型改进是否成功",
        "",
        "探索方法在Deep Lasso的预测损失与输入梯度稀疏惩罚之外，增加同月互补卖家子群的归一化输入梯度分布距离。正则强度只根据训练—调参期选择。",
        "",
        markdown(pd.DataFrame(tuning["trials"])[[
            "gamma", "tune_rmse_log", "tune_seller_half_jaccard",
            "tune_month_jaccard",
        ]]),
        "",
        f"预设规则最终选择gamma={tuning['chosen_gamma']}。这意味着新增一致性惩罚没有提供可识别的开发期收益，应当淘汰该改进，而不是改用测试集挑参数。",
        "",
        markdown(pd.DataFrame([
            {
                "方法": "探索训练流程（gamma=0）",
                "XGBoost测试RMSE": pilot_prediction["xgboost"]["test_rmse"],
                "TabM测试RMSE": pilot_prediction["tabm"]["test_rmse"],
            },
            {
                "方法": "原Deep Lasso",
                "XGBoost测试RMSE": v7_prediction.loc[
                    v7_prediction.method.eq("deep_lasso")
                    & v7_prediction.evaluator.eq("xgboost")
                    & v7_prediction.k.eq(K), "rmse_log"
                ].iloc[0],
                "TabM测试RMSE": v7_prediction.loc[
                    v7_prediction.method.eq("deep_lasso")
                    & v7_prediction.evaluator.eq("tabm")
                    & v7_prediction.k.eq(K), "rmse_log"
                ].iloc[0],
            },
            {
                "方法": "XGBoost-SHAP",
                "XGBoost测试RMSE": v7_prediction.loc[
                    v7_prediction.method.eq("xgboost_shap")
                    & v7_prediction.evaluator.eq("xgboost")
                    & v7_prediction.k.eq(K), "rmse_log"
                ].iloc[0],
                "TabM测试RMSE": v7_prediction.loc[
                    v7_prediction.method.eq("xgboost_shap")
                    & v7_prediction.evaluator.eq("tabm")
                    & v7_prediction.k.eq(K), "rmse_log"
                ].iloc[0],
            },
        ])),
        "",
        "测试期早已在V7中被查看，上表仅作后验描述。由于开发期门槛已经淘汰gamma>0，没有继续为失败正则运行完整60个半样本和模拟，也没有将gamma=0重新包装成新方法。",
        "",
        "## 2. 变量质量审计",
        "",
        "本审计只使用开发期滚动月份，不能把分类解释为变量为真的概率或因果效应。",
        "",
        "### 透明准入规则",
        "",
        "- 核心候选：四个强方法中至少三个选中；半样本平均频率不低于0.70；从31维全模型删除后至少60%的月份误差变差；历史月份选择的最佳替代变量在后续月份平均仍不如原变量。",
        "- 可替代候选：至少两个强方法选中，但与其他变量高度相关，或历史最优替代在后续月份不劣于原变量。",
        "- 观察候选：有部分预测或选择证据，但尚未同时通过上述要求。",
        "- 证据不足：当前窗口没有形成足够一致的增量证据。",
        "",
        "### 31项结果",
        "",
        markdown(summary),
        "",
        "### 分类规则敏感性",
        "",
        "原严格规则对各选择器替代损失取均值。检查后发现，某个基线名单若遗漏一个强变量，会使多项替换同时改善，从而主导均值。为透明呈现这一问题，另报告跨选择器替代损失中位数与多数方向规则；两种分类均保留，后者不是预先确认结果。",
        "",
        markdown(sensitivity),
        "",
        "### 分选择器的删除与替代结果",
        "",
        markdown(detail),
        "",
        "## 3. 组感知多解准入的后验试验",
        "",
        "仅依赖不可替代核心变量会丢失大量联合预测信息。因此构造组感知名单：以绝对Spearman相关不低于0.80建立相关图，每个连通组先保留一个代表，再按强方法共识数、半样本频率、滚动月份删除方向和删除损失作词典序排序，不把指标任意加权成总分。",
        "",
        markdown(set_prediction.loc[
            set_prediction["set"].isin([
                "group_aware_top8", "group_aware_top12",
                "xgboost_shap_top8", "xgboost_shap_top12",
                "deep_lasso_top8", "deep_lasso_top12",
                "all_features",
            ]),
            ["set", "k", "evaluator", "features",
             "validation_rmse", "test_rmse_posthoc"],
        ]),
        "",
        markdown(pd.DataFrame([
            {
                "evaluator": evaluator,
                "group_minus_shap_rmse": values["point_difference"],
                "ci95_low": values["ci95_low"],
                "ci95_high": values["ci95_high"],
            }
            for evaluator, values in paired.items()
            if evaluator in ("xgboost", "tabm")
        ])),
        "",
        "组感知Top-8的测试RMSE没有优于SHAP Top-8，但两种评价器的配对区间均跨零。组感知Top-8的平均两两绝对Spearman相关为0.146，且不存在相关不低于0.80的变量对；SHAP Top-8分别为0.191和3对。它体现的是以很小预测差异换取明显去冗余，而不是预测优势。",
        "",
        "该规则是在V7结果及本次变量审计之后形成，属于后验方法原型。正式论文采用前必须在新时间窗口或第二数据集冻结后验证，并在每个重抽样样本内完整重算规则，才能评价其选择稳定性。",
    ]
    (PILOT / "变量质量审计.md").write_text("\n".join(lines))


def evaluate_audit_sets():
    destination = RESULTS / "audit_set_prediction_v2.csv"
    if destination.exists():
        return
    sensitivity = pd.read_csv(
        RESULTS / "variable_quality_rule_sensitivity.csv"
    )
    summary = pd.read_csv(
        RESULTS / "variable_quality_summary.csv"
    ).set_index("label_zh")
    core = sensitivity.loc[
        sensitivity.robust_majority_category.eq("核心候选"), "label_zh"
    ].tolist()
    robust_observation = sensitivity.loc[
        sensitivity.robust_majority_category.eq("观察候选"), "label_zh"
    ].tolist()
    strong_observation = [
        label for label in robust_observation
        if summary.loc[label, "selected_by_strong_methods_0_to_4"] >= 3
    ]
    dictionary = pd.read_csv(ROOT / "results/feature_dictionary.csv")
    label_to_index = {
        row.label_zh: index for index, row in dictionary.iterrows()
    }
    sets = {
        "audit_core": [label_to_index[label] for label in core],
        "audit_core_plus_strong_observation": [
            label_to_index[label] for label in core + strong_observation
        ],
    }
    frame = load_data()
    correlation = frame[FEATURES].corr(method="spearman").abs()
    remaining, components = set(FEATURES), []
    while remaining:
        start = remaining.pop()
        stack, component = [start], []
        while stack:
            current = stack.pop()
            component.append(current)
            adjacent = [
                item for item in list(remaining)
                if correlation.loc[current, item] >= .80
            ]
            for item in adjacent:
                remaining.remove(item)
                stack.append(item)
        components.append(component)
    component_id = {
        feature: index
        for index, component in enumerate(components)
        for feature in component
    }
    ranked = pd.read_csv(
        RESULTS / "variable_quality_summary.csv"
    ).set_index("feature").sort_values(
        [
            "selected_by_strong_methods_0_to_4",
            "mean_half_sample_frequency",
            "full_model_drop_positive_month_fraction",
            "full_model_drop_delta_mean",
        ],
        ascending=False,
    )
    group_order, used_components = [], set()
    for feature in ranked.index:
        group = component_id[feature]
        if group not in used_components:
            group_order.append(int(ranked.loc[feature, "feature_index"]))
            used_components.add(group)
    for feature in ranked.index:
        index = int(ranked.loc[feature, "feature_index"])
        if index not in group_order:
            group_order.append(index)
    sets["group_aware_top8"] = group_order[:8]
    sets["group_aware_top12"] = group_order[:12]
    for method in ("xgboost_shap", "deep_lasso"):
        order = read(ROOT / f"results/runs/{method}/reference.json")["order"]
        for size in (3, 5, 8, 12):
            sets[f"{method}_top{size}"] = order[:size]
    sets["all_features"] = list(range(len(FEATURES)))
    train = frame.loc[frame.split.isin(["train", "tune"])]
    valid = frame.loc[frame.split.eq("rank")]
    test = frame.loc[frame.split.eq("test")]
    pre = Preprocessor().fit(train[FEATURES].to_numpy(float))
    xall = pre.transform(train[FEATURES].to_numpy(float))
    xvall = pre.transform(valid[FEATURES].to_numpy(float))
    xtall = pre.transform(test[FEATURES].to_numpy(float))
    y = train.log_gmv_next_month.to_numpy(float)
    yv = valid.log_gmv_next_month.to_numpy(float)
    yt = test.log_gmv_next_month.to_numpy(float)
    rows = []
    for set_name, columns in sets.items():
        for evaluator, config_name in (
            ("xgboost", "xgboost_shap"), ("tabm", "tabm")
        ):
            config = read(
                ROOT / f"results/configs/{config_name}.json"
            )["params"]
            valid_predictions, test_predictions = [], []
            for seed in SEEDS:
                x, xv, xt = (
                    xall[:, columns], xvall[:, columns], xtall[:, columns]
                )
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
                    best_epoch = min(
                        meta["history"], key=lambda row: row[2]
                    )[0]
                    model, _ = fit_neural(
                        np.concatenate([x, xv]), np.r_[y, yv], xv, yv,
                        config | {
                            "epochs": best_epoch, "fixed_epochs": True
                        },
                        "tabm", seed,
                    )
                    test_predictions.append(neural_predict(model, xt))
            rows.append({
                "set": set_name, "k": len(columns), "evaluator": evaluator,
                "features": "|".join(FEATURES[j] for j in columns),
                "validation_rmse": float(mean_squared_error(
                    yv, np.mean(valid_predictions, axis=0)
                ) ** .5),
                "test_rmse_posthoc": float(mean_squared_error(
                    yt, np.mean(test_predictions, axis=0)
                ) ** .5),
                "test_previously_observed": True,
            })
            print(set_name, evaluator, rows[-1]["test_rmse_posthoc"])
    pd.DataFrame(rows).to_csv(destination, index=False)


def paired_group_audit():
    destination = RESULTS / "group_aware_top8_paired_intervals.json"
    if destination.exists():
        return
    sets_frame = pd.read_csv(RESULTS / "audit_set_prediction_v2.csv")
    sets = {}
    for name in ("group_aware_top8", "xgboost_shap_top8"):
        encoded = sets_frame.loc[
            sets_frame["set"].eq(name)
            & sets_frame.evaluator.eq("xgboost"), "features"
        ].iloc[0]
        sets[name] = [FEATURES.index(item) for item in encoded.split("|")]
    frame = load_data()
    train = frame.loc[frame.split.isin(["train", "tune"])]
    valid = frame.loc[frame.split.eq("rank")]
    test = frame.loc[frame.split.eq("test")]
    pre = Preprocessor().fit(train[FEATURES].to_numpy(float))
    xall = pre.transform(train[FEATURES].to_numpy(float))
    xvall = pre.transform(valid[FEATURES].to_numpy(float))
    xtall = pre.transform(test[FEATURES].to_numpy(float))
    y = train.log_gmv_next_month.to_numpy(float)
    yv = valid.log_gmv_next_month.to_numpy(float)
    yt = test.log_gmv_next_month.to_numpy(float)
    predictions = {}
    for evaluator, config_name in (
        ("xgboost", "xgboost_shap"), ("tabm", "tabm")
    ):
        config = read(ROOT / f"results/configs/{config_name}.json")["params"]
        predictions[evaluator] = {}
        for name, columns in sets.items():
            rows = []
            x, xv, xt = (
                xall[:, columns], xvall[:, columns], xtall[:, columns]
            )
            for seed in SEEDS:
                if evaluator == "xgboost":
                    model = xgb_model(config, seed).fit(
                        np.concatenate([x, xv]), np.r_[y, yv]
                    )
                    rows.append(model.predict(xt))
                else:
                    model, meta = fit_neural(
                        x, y, xv, yv, config, "tabm", seed
                    )
                    best_epoch = min(
                        meta["history"], key=lambda row: row[2]
                    )[0]
                    model, _ = fit_neural(
                        np.concatenate([x, xv]), np.r_[y, yv], xv, yv,
                        config | {
                            "epochs": best_epoch, "fixed_epochs": True
                        },
                        "tabm", seed,
                    )
                    rows.append(neural_predict(model, xt))
            predictions[evaluator][name] = np.mean(rows, axis=0)
    sellers, inverse = np.unique(test.seller_id, return_inverse=True)
    sizes = np.bincount(inverse)
    draws = np.random.default_rng(916).integers(
        0, len(sellers), (2000, len(sellers))
    )
    output = {}
    for evaluator in predictions:
        errors = {
            name: (prediction - yt) ** 2
            for name, prediction in predictions[evaluator].items()
        }
        bootstrap = {}
        for name, values in errors.items():
            sums = np.bincount(inverse, weights=values)
            bootstrap[name] = np.sqrt(
                sums[draws].sum(1) / sizes[draws].sum(1)
            )
        difference = (
            bootstrap["group_aware_top8"]
            - bootstrap["xgboost_shap_top8"]
        )
        output[evaluator] = {
            "difference_definition": "group-aware RMSE minus XGBoost-SHAP RMSE",
            "point_difference": float(
                np.sqrt(errors["group_aware_top8"].mean())
                - np.sqrt(errors["xgboost_shap_top8"].mean())
            ),
            "ci95_low": float(np.quantile(difference, .025)),
            "ci95_high": float(np.quantile(difference, .975)),
            "seller_cluster_bootstrap_repetitions": 2000,
        }
    output.update({
        "conditional_on_observed_three_test_months": True,
        "existing_test_previously_observed": True,
        "post_v7_exploratory": True,
    })
    save_json(destination, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["run", "report", "evaluate", "paired"]
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        run()
    elif args.stage == "report":
        report()
    elif args.stage == "evaluate":
        evaluate_audit_sets()
    else:
        paired_group_audit()
