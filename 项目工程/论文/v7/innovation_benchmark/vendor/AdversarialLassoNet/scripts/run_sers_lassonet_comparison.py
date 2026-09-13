import argparse
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths

add_project_src_paths(ROOT)

from adversarial_lassonet.paths import OUTPUTS_ROOT, ensure_directory, get_data_root

from lassonet import LassoNetClassifier
from models.lassonet_sers.lassonet_sam_input import LassoNetSAMInputClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed fixed: {seed}")


def load_raw_data(data_dir: Path | None = None):
    print("Loading data...")
    sers_dir = data_dir or (get_data_root() / "sers")
    healthy_path = sers_dir / "HealthyControl0.csv"
    cancer_path = sers_dir / "LungCancer0.csv"
    if not healthy_path.exists() or not cancer_path.exists():
        raise FileNotFoundError(
            "SERS CSV files not found. Expected:\n"
            f"  {healthy_path}\n"
            f"  {cancer_path}\n"
            "Please place the files under data/raw/sers/ or set LASSONET_DATA_DIR."
        )
    df_h = pd.read_csv(healthy_path, index_col=0).dropna()
    df_c = pd.read_csv(cancer_path, index_col=0).dropna()

    feat_h = df_h.drop(columns=["Label"]) if "Label" in df_h.columns else df_h
    feat_c = df_c.drop(columns=["Label"]) if "Label" in df_c.columns else df_c

    n_features = min(feat_h.shape[1], feat_c.shape[1])
    x = np.vstack(
        [
            feat_h.iloc[:, :n_features].values,
            feat_c.iloc[:, :n_features].values,
        ]
    )
    y = np.hstack([np.zeros(len(feat_h)), np.ones(len(feat_c))]).astype(int)
    return x, y


def split_and_scale(x, y, random_state=42):
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    print(
        "Split sizes: "
        f"train={len(x_train)}, val={len(x_val)}, test={len(x_test)}"
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def get_model_for_eval(algo):
    for name in ("model", "net", "_model"):
        model = getattr(algo, name, None)
        if model is not None:
            return model
    return None


def build_model(kind, adv_rho, adv_alpha, adv_delta):
    common_kwargs = dict(
        hidden_dims=(2048, 1024, 512),
        M=10,
        verbose=True,
        lambda_start=0.001,
        path_multiplier=1.1,
        optim=(
            partial(torch.optim.Adam, lr=1e-5),
            partial(torch.optim.SGD, lr=1e-5, momentum=0.9),
        ),
        n_iters=300,
        val_size=0,
    )

    if kind == "clean":
        return LassoNetClassifier(**common_kwargs)
    if kind == "sam":
        return LassoNetSAMInputClassifier(
            **common_kwargs,
            adv_rho=adv_rho,
            adv_alpha=adv_alpha,
            adv_delta=adv_delta,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def evaluate_binary_screening(model, x, y, split_name):
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    return {
        f"{split_name}_accuracy": accuracy_score(y, y_pred),
        f"{split_name}_sensitivity": sensitivity,
        f"{split_name}_specificity": specificity,
        f"{split_name}_auc": roc_auc_score(y, y_proba),
        f"{split_name}_precision_ppv": precision_score(y, y_pred, zero_division=0),
        f"{split_name}_recall": recall_score(y, y_pred, zero_division=0),
        f"{split_name}_f1": f1_score(y, y_pred, zero_division=0),
        f"{split_name}_npv": npv,
        f"{split_name}_balanced_accuracy": balanced_acc,
        f"{split_name}_tn": int(tn),
        f"{split_name}_fp": int(fp),
        f"{split_name}_fn": int(fn),
        f"{split_name}_tp": int(tp),
    }


def print_screening_metrics(kind, metrics, split_name):
    print(f"\n{kind} {split_name} screening metrics:")
    print(
        f"  Accuracy:    {metrics[f'{split_name}_accuracy']:.4f}\n"
        f"  Sensitivity: {metrics[f'{split_name}_sensitivity']:.4f}\n"
        f"  Specificity: {metrics[f'{split_name}_specificity']:.4f}\n"
        f"  AUC:         {metrics[f'{split_name}_auc']:.4f}\n"
        f"  PPV:         {metrics[f'{split_name}_precision_ppv']:.4f}\n"
        f"  NPV:         {metrics[f'{split_name}_npv']:.4f}\n"
        f"  F1:          {metrics[f'{split_name}_f1']:.4f}\n"
        f"  BalancedAcc: {metrics[f'{split_name}_balanced_accuracy']:.4f}\n"
        f"  Confusion:   TN={metrics[f'{split_name}_tn']}, "
        f"FP={metrics[f'{split_name}_fp']}, "
        f"FN={metrics[f'{split_name}_fn']}, "
        f"TP={metrics[f'{split_name}_tp']}"
    )


def train_one(kind, x_train, x_val, x_test, y_train, y_val, y_test, args):
    print("\n" + "=" * 70)
    print(f"Start training: {kind}")
    print("=" * 70)
    set_seed(args.seed)
    if kind == "sam":
        print(
            "SAM settings aligned with adv.py: "
            f"adv_rho={args.adv_rho}, "
            f"adv_alpha={args.adv_alpha}, "
            f"adv_delta={args.adv_delta}"
        )

    model = build_model(kind, args.adv_rho, args.adv_alpha, args.adv_delta)
    history_feats = []
    history_val_accs = []
    all_models_info = []
    best_val_acc = 0.0
    best_model_state = None
    best_lambda = None
    best_selected_features = 0

    def monitor(algo, history):
        nonlocal best_val_acc, best_model_state, best_lambda, best_selected_features

        model_ref = get_model_for_eval(algo)
        was_training = model_ref.training if model_ref is not None else None

        state = history[-1]
        mask = state.selected
        n_selected = mask.sum().item() if torch.is_tensor(mask) else np.sum(mask)

        if model_ref is not None:
            model_ref.eval()
        val_acc = algo.score(x_val, y_val)
        if model_ref is not None and was_training:
            model_ref.train()

        history_feats.append(n_selected)
        history_val_accs.append(val_acc)
        all_models_info.append(
            {
                "lambda": state.lambda_,
                "val_acc": val_acc,
                "n_selected": n_selected,
                "state_dict": state.state_dict,
            }
        )

        if len(history) % 10 == 0 or n_selected < 300:
            print(f"{kind:5s} | Features: {n_selected:4d} | Val Acc: {val_acc:.4f}")

        if n_selected < 300 and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = state.state_dict
            best_lambda = state.lambda_
            best_selected_features = n_selected
            print(
                f"{kind:5s} | New best: "
                f"lambda={best_lambda:.6f}, features={n_selected}, val={val_acc:.4f}"
            )

    path = model.path(
        x_train,
        y_train,
        X_val=x_val,
        y_val=y_val,
        callback=monitor,
        return_state_dicts=True,
    )

    if best_model_state is None:
        candidates = [m for m in all_models_info if m["n_selected"] < 300]
        if not candidates:
            print(f"{kind}: no model found with <300 features.")
            return {
                "kind": kind,
                "ok": False,
                "history_feats": history_feats,
                "history_val_accs": history_val_accs,
            }

        best_candidate = max(candidates, key=lambda item: item["val_acc"])
        for item in path:
            if item.state_dict is not None and abs(item.lambda_ - best_candidate["lambda"]) < 1e-10:
                best_model_state = item.state_dict
                best_lambda = item.lambda_
                best_selected_features = best_candidate["n_selected"]
                best_val_acc = best_candidate["val_acc"]
                break

    model.load(best_model_state)
    val_metrics = evaluate_binary_screening(model, x_val, y_val, "val")
    test_metrics = evaluate_binary_screening(model, x_test, y_test, "test")
    test_acc = test_metrics["test_accuracy"]

    global_best_idx = int(np.argmax(history_val_accs))
    result = {
        "kind": kind,
        "ok": True,
        "best_lambda": best_lambda,
        "best_selected_features": best_selected_features,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "global_best_val_acc": history_val_accs[global_best_idx],
        "global_best_features": history_feats[global_best_idx],
        "history_feats": history_feats,
        "history_val_accs": history_val_accs,
    }
    result.update(val_metrics)
    result.update(test_metrics)

    print(f"{kind}: best <300 features val acc = {best_val_acc:.4f}")
    print(f"{kind}: best features = {best_selected_features}")
    print(f"{kind}: best lambda = {best_lambda:.6f}")
    print(f"{kind}: test acc = {test_acc:.4f}")
    print_screening_metrics(kind, val_metrics, "val")
    print_screening_metrics(kind, test_metrics, "test")
    return result


def plot_results(results, output_path):
    plt.figure(figsize=(10, 6))
    colors = {"clean": "tab:blue", "sam": "tab:orange"}
    for result in results:
        if not result["history_feats"]:
            continue
        plt.plot(
            result["history_feats"],
            result["history_val_accs"],
            "o-",
            markersize=3,
            label=result["kind"],
            color=colors.get(result["kind"]),
        )
    plt.gca().invert_xaxis()
    plt.xlabel("Number of Features")
    plt.ylabel("Validation Accuracy")
    plt.title("Task 1 Comparison: Clean LassoNet vs SAM Input")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlot saved as {output_path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("Comparison summary")
    print("=" * 70)
    print(
        f"{'Model':<8} {'ValAcc':>8} {'TestAcc':>8} {'Sens':>8} "
        f"{'Spec':>8} {'AUC':>8} {'PPV':>8} {'NPV':>8} {'F1':>8} "
        f"{'Features':>10}"
    )
    for result in results:
        if not result["ok"]:
            print(f"{result['kind']:<8} failed")
            continue
        print(
            f"{result['kind']:<8} "
            f"{result['val_accuracy']:>8.4f} "
            f"{result['test_accuracy']:>8.4f} "
            f"{result['test_sensitivity']:>8.4f} "
            f"{result['test_specificity']:>8.4f} "
            f"{result['test_auc']:>8.4f} "
            f"{result['test_precision_ppv']:>8.4f} "
            f"{result['test_npv']:>8.4f} "
            f"{result['test_f1']:>8.4f} "
            f"{result['best_selected_features']:>10}"
        )

    valid = {result["kind"]: result for result in results if result["ok"]}
    if "clean" in valid and "sam" in valid:
        print("\nDifferences: sam - clean")
        print(f"Val Acc diff:  {valid['sam']['best_val_acc'] - valid['clean']['best_val_acc']:+.4f}")
        print(f"Test Acc diff: {valid['sam']['test_accuracy'] - valid['clean']['test_accuracy']:+.4f}")
        print(
            "Sensitivity diff: "
            f"{valid['sam']['test_sensitivity'] - valid['clean']['test_sensitivity']:+.4f}"
        )
        print(
            "Specificity diff: "
            f"{valid['sam']['test_specificity'] - valid['clean']['test_specificity']:+.4f}"
        )
        print(f"AUC diff:      {valid['sam']['test_auc'] - valid['clean']['test_auc']:+.4f}")
        print(
            "Feature diff:  "
            f"{valid['sam']['best_selected_features'] - valid['clean']['best_selected_features']:+d}"
        )


def save_metrics_csv(results, output_path):
    rows = []
    for result in results:
        if not result["ok"]:
            continue
        row = {
            "model": result["kind"],
            "best_lambda": result["best_lambda"],
            "best_selected_features": result["best_selected_features"],
            "best_val_acc_under300": result["best_val_acc"],
            "global_best_val_acc": result["global_best_val_acc"],
        }
        for split_name in ("val", "test"):
            for metric_name in (
                "accuracy",
                "sensitivity",
                "specificity",
                "auc",
                "precision_ppv",
                "npv",
                "f1",
                "balanced_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ):
                key = f"{split_name}_{metric_name}"
                row[key] = result[key]
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Metrics saved as {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare clean LassoNet and SAM-input LassoNet on the SERS task."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "clean", "sam"],
        default="both",
        help="Which model to run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        default=str(OUTPUTS_ROOT / "exp1_main_benchmark" / "LassoNet_Task1_compare.png"),
        help="Output path for the comparison plot.",
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(
            OUTPUTS_ROOT / "exp1_main_benchmark" / "LassoNet_Task1_compare_metrics.csv"
        ),
        help="Output CSV path for screening metrics.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing HealthyControl0.csv and LungCancer0.csv.",
    )
    parser.add_argument("--adv-rho", type=float, default=0.1)
    parser.add_argument("--adv-alpha", type=float, default=1.0)
    parser.add_argument("--adv-delta", type=float, default=1e-12)
    return parser


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    set_seed(args.seed)
    plot_path = Path(args.plot)
    metrics_csv_path = Path(args.metrics_csv)
    ensure_directory(plot_path.parent)
    ensure_directory(metrics_csv_path.parent)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    x, y = load_raw_data(data_dir=data_dir)
    print(f"Input feature dim: {x.shape[1]}")
    print(f"Total samples: {len(x)}")
    data = split_and_scale(x, y, random_state=args.seed)

    kinds = ["clean", "sam"] if args.mode == "both" else [args.mode]
    results = [train_one(kind, *data, args) for kind in kinds]
    print_summary(results)
    save_metrics_csv(results, metrics_csv_path)
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()
