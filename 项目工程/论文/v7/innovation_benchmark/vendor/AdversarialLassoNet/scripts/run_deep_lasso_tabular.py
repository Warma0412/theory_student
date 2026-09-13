import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from models.deep_lasso_tabular import DeepLassoTabularPipeline
from utils.data_utils import load_dataset
DATASET_ALIAS = {
    "mnist": "MNIST",
    "minst": "MNIST",
    "coloredmnist": "ColoredMNIST",
    "colored-mnist": "ColoredMNIST",
    "colored_mnist": "ColoredMNIST",
    "mnist-fashion": "MNIST-Fashion",
    "fashion": "MNIST-Fashion",
    "mice": "MICE",
    "coil": "COIL",
    "isolet": "ISOLET",
    "activity": "Activity",
}

VALID_DATASETS = [
    "MNIST",
    "ColoredMNIST",
    "MNIST-Fashion",
    "MICE",
    "COIL",
    "ISOLET",
    "Activity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabular Deep-Lasso runner backed by the extracted training pipeline"
    )
    parser.add_argument("--dataset", type=str, default="MICE")
    parser.add_argument("--k", type=int, default=50, help="Target selected feature count.")
    parser.add_argument("--n-runs", "--runs", dest="n_runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--refit-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=0.2)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc"],
        help="Checkpoint selection metric during regularized training.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-dir", type=str, default=".")
    return parser.parse_args()


def normalize_dataset_name(dataset: str) -> str:
    return DATASET_ALIAS.get(dataset.lower(), dataset)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_ready(summary: dict) -> dict:
    ans = dict(summary)
    for key in ["selected_mask", "importance", "selector_state_dict", "refit_state_dict"]:
        value = ans.pop(key, None)
        if key == "selected_mask" and value is not None:
            ans["selected_mask"] = value.astype(bool).tolist()
        if key == "importance" and value is not None:
            ans["importance"] = value.astype(float).tolist()
    return ans


def run_once(
    pipeline: DeepLassoTabularPipeline,
    dataset: str,
    seed: int,
    k: int,
):
    loaded = load_dataset(dataset)
    if loaded is None:
        valid = ", ".join(VALID_DATASETS)
        raise ValueError(f"Invalid dataset '{dataset}'. Valid options: {valid}")

    (X_train_valid, y_train_valid), (X_test, y_test) = loaded
    X_train_valid = X_train_valid.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train_valid = y_train_valid.astype(np.int64)
    y_test = y_test.astype(np.int64)

    result = pipeline.run(
        dataset=dataset,
        seed=seed,
        X_train_valid=X_train_valid,
        y_train_valid=y_train_valid,
        X_test=X_test,
        y_test=y_test,
        k=k,
    )
    return result.__dict__


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DeepLassoTabularPipeline(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        epochs=args.epochs,
        refit_epochs=args.refit_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reg_weight=args.reg_weight,
        selection_metric=args.selection_metric,
        device=device,
    )

    print(f"Device: {device}")
    print(
        f"Config: dataset={dataset}, runs={args.n_runs}, k={args.k}, epochs={args.epochs}, "
        f"refit_epochs={args.refit_epochs}, batch_size={args.batch_size}, reg_weight={args.reg_weight}"
    )

    run_summaries = []
    selected_counts = []
    test_scores = []

    for run_idx in range(args.n_runs):
        current_seed = args.base_seed + run_idx
        print(f"\n===== Run {run_idx + 1}/{args.n_runs} (seed={current_seed}) =====")
        summary = run_once(pipeline, dataset, current_seed, args.k)
        run_summaries.append(json_ready(summary))
        selected_counts.append(summary["selected_count"])
        test_scores.append(summary["test_score"])
        print("Number of selected features:", summary["selected_count"])
        print("Selector best val acc:", summary["selector_best_val_acc"])
        print("Refit val score:", summary["refit_val_score"])
        print("Test score:", summary["test_score"])

        payload = {
            "dataset": dataset,
            "seed": current_seed,
            "method": "deep_lasso",
            "selected_mask": summary["selected_mask"].astype(bool),
            "selected_count": summary["selected_count"],
            "selected_indices": summary["selected_indices"],
            "importance": summary["importance"].astype(np.float32),
            "selector_state_dict": summary["selector_state_dict"],
            "refit_state_dict": summary["refit_state_dict"],
            "config": {
                "k": args.k,
                "epochs": args.epochs,
                "refit_epochs": args.refit_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "reg_weight": args.reg_weight,
                "depth": args.depth,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "selection_metric": args.selection_metric,
            },
        }
        with open(save_dir / f"{dataset}_deep_lasso_path_seed{current_seed}.pkl", "wb") as f:
            pickle.dump(payload, f)

    count_std = float(np.std(selected_counts, ddof=1)) if len(selected_counts) > 1 else 0.0
    score_std = float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0
    aggregate = {
        "dataset": dataset,
        "method": "deep_lasso",
        "k": args.k,
        "n_runs": args.n_runs,
        "base_seed": args.base_seed,
        "selected_count_mean": float(np.mean(selected_counts)),
        "selected_count_std": count_std,
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": score_std,
        "runs": run_summaries,
    }

    summary_path = save_dir / f"{dataset}_deep_lasso_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(
        f"Selected features mean +- std: {aggregate['selected_count_mean']:.2f} +- "
        f"{aggregate['selected_count_std']:.2f}"
    )
    print(
        f"Test score mean +- std: {aggregate['test_score_mean']:.6f} +- "
        f"{aggregate['test_score_std']:.6f}"
    )
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
