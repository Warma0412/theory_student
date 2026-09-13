import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths

add_project_src_paths(ROOT)

from adversarial_lassonet.paths import OUTPUTS_ROOT

from adversarial_lassonet import (  
    AdversarialLassoNetClassifier,
    canonical_dataset_name,
    resolve_device,
    set_seed,
)
from models.deep_lasso_tabular import DeepLassoTabularPipeline
from models.fista_tabular import FISTATabularClassifier  
from utils.data_utils import load_dataset 
from lassonet import LassoNetClassifier  
from lassonet.utils import eval_on_path  


TABLE2_DATASETS = ["MICE", "MNIST", "MNIST-Fashion", "ISOLET", "COIL", "Activity"]
METHOD_ORDER = ["vanilla_lassonet", "fista_net", "deep_lasso", "proposed_method"]
METHOD_LABELS = {
    "vanilla_lassonet": "Vanilla LassoNet",
    "fista_net": "FISTA-Net",
    "deep_lasso": "Deep-Lasso",
    "proposed_method": "Proposed Method",
}
DISPLAY_DATASETS = {
    "MICE": "Mice",
    "MNIST": "MNIST",
    "MNIST-Fashion": "MNIST-Fashion",
    "ISOLET": "ISOLET",
    "COIL": "COIL-20",
    "Activity": "Activity",
}


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    data_dim: int


@dataclass
class MethodResult:
    selected_count: int
    val_score: float
    test_score: float
    selected_lambda: float | None


def parse_benchmark_args(return_parser: bool = False):
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for Table 2-style comparison."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="MICE,MNIST,MNIST-Fashion,ISOLET,COIL,Activity",
        help='Comma-separated dataset list or "table2".',
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lasso-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lasso-verbose", type=int, default=0)
    parser.add_argument("--fista-batch-size", type=int, default=256)
    parser.add_argument("--fista-n-iters-init", type=int, default=200)
    parser.add_argument("--fista-n-iters-path", type=int, default=100)
    parser.add_argument("--fista-n-iters-refit", type=int, default=300)
    parser.add_argument("--fista-lambda-start", type=float, default=1e-4)
    parser.add_argument("--fista-path-multiplier", type=float, default=1.5)
    parser.add_argument("--fista-lr", type=float, default=1e-3)
    parser.add_argument("--fista-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fista-verbose", type=int, default=0)
    parser.add_argument("--deep-lasso-epochs", type=int, default=200)
    parser.add_argument("--deep-lasso-refit-epochs", type=int, default=200)
    parser.add_argument("--deep-lasso-hidden-dim", type=int, default=0)
    parser.add_argument("--deep-lasso-depth", type=int, default=2)
    parser.add_argument("--deep-lasso-dropout", type=float, default=0.0)
    parser.add_argument("--deep-lasso-lr", type=float, default=1e-3)
    parser.add_argument("--deep-lasso-weight-decay", type=float, default=1e-4)
    parser.add_argument("--deep-lasso-reg-weight", type=float, default=0.2)
    parser.add_argument(
        "--deep-lasso-selection-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc"],
    )
    parser.add_argument("--adv-rho", type=float, default=0.1)
    parser.add_argument("--adv-alpha", type=float, default=1.0)
    parser.add_argument("--adv-delta", type=float, default=1e-12)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--grad-norm-weight", type=float, default=0.1)
    parser.add_argument("--prox-lambda-bar-ratio", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_ROOT / "exp1_main_benchmark"),
    )
    parser.add_argument("--output-prefix", type=str, default="table2")
    if return_parser:
        return parser
    return parser.parse_args()


def resolve_benchmark_datasets(text: str) -> List[str]:
    if text.lower() == "table2":
        return TABLE2_DATASETS
    return [canonical_dataset_name(token.strip()) for token in text.split(",") if token.strip()]


def load_split(dataset: str, seed: int) -> DatasetSplit:
    loaded = load_dataset(dataset)
    if loaded is None:
        raise ValueError(f"Invalid dataset: {dataset}")

    (X_train_valid, y_train_valid), (X_test, y_test) = loaded
    y_train_valid = y_train_valid.astype(np.int64)
    y_test = y_test.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.125,
        random_state=seed,
        stratify=y_train_valid if len(np.unique(y_train_valid)) > 1 else None,
    )
    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        data_dim=X_test.shape[1],
    )


def build_lassonet(input_dim: int, seed: int, args: argparse.Namespace, device: torch.device):
    hidden_dim = (max(1, input_dim // 3),)
    return LassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=args.lasso_epochs,
        batch_size=args.batch_size,
    )


def choose_target_checkpoint(path: Iterable, target_k: int):
    return min(path, key=lambda save: abs(int(save.selected.sum().item()) - target_k))


def choose_nonempty_fista_checkpoint(path: Iterable, target_k: int):
    nonempty = [save for save in path if int(save.selected.sum().item()) > 0]
    if not nonempty:
        raise RuntimeError(
            "FISTA path selected 0 features for every checkpoint. "
            "Try a smaller --fista-lambda-start or --fista-path-multiplier."
        )
    return min(nonempty, key=lambda save: abs(int(save.selected.sum().item()) - target_k))


def run_vanilla_lassonet(
    split: DatasetSplit, seed: int, args: argparse.Namespace, device: torch.device
) -> MethodResult:
    selector = build_lassonet(split.X_train.shape[1], seed, args, device)
    path = selector.path(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
    desired = choose_target_checkpoint(path, args.k)
    selected_mask = desired.selected.cpu().numpy().astype(bool)
    selected_count = int(selected_mask.sum())
    if selected_count == 0:
        raise RuntimeError("Vanilla LassoNet selected 0 features.")

    X_train_selected = split.X_train[:, selected_mask]
    X_val_selected = split.X_val[:, selected_mask]
    X_test_selected = split.X_test[:, selected_mask]

    refit = build_lassonet(X_train_selected.shape[1], seed, args, device)
    path_sparse = refit.path(
        X_train_selected,
        split.y_train,
        X_val=X_val_selected,
        y_val=split.y_val,
        return_state_dicts=True,
    )[:1]
    val_score = float(eval_on_path(refit, path_sparse, X_val_selected, split.y_val)[0])
    test_score = float(eval_on_path(refit, path_sparse, X_test_selected, split.y_test)[0])
    return MethodResult(
        selected_count=selected_count,
        val_score=val_score,
        test_score=test_score,
        selected_lambda=float(desired.lambda_),
    )


def run_deep_lasso(
    split: DatasetSplit, seed: int, args: argparse.Namespace, device: torch.device
) -> MethodResult:
    pipeline = DeepLassoTabularPipeline(
        hidden_dim=args.deep_lasso_hidden_dim,
        depth=args.deep_lasso_depth,
        dropout=args.deep_lasso_dropout,
        epochs=args.deep_lasso_epochs,
        refit_epochs=args.deep_lasso_refit_epochs,
        batch_size=args.batch_size,
        lr=args.deep_lasso_lr,
        weight_decay=args.deep_lasso_weight_decay,
        reg_weight=args.deep_lasso_reg_weight,
        selection_metric=args.deep_lasso_selection_metric,
        device=device,
    )
    result = pipeline.run(
        dataset="benchmark",
        seed=seed,
        X_train_valid=np.concatenate((split.X_train, split.X_val), axis=0),
        y_train_valid=np.concatenate((split.y_train, split.y_val), axis=0),
        X_test=split.X_test,
        y_test=split.y_test,
        k=args.k,
    )
    return MethodResult(
        selected_count=int(result.selected_count),
        val_score=float(result.refit_val_score),
        test_score=float(result.test_score),
        selected_lambda=None,
    )


def run_proposed_method(
    split: DatasetSplit, seed: int, args: argparse.Namespace, device: torch.device
) -> MethodResult:
    hidden_dim = (max(1, split.X_train.shape[1] // 3),)
    selector = AdversarialLassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=args.lasso_epochs,
        batch_size=args.batch_size,
        adv_rho=args.adv_rho,
        adv_alpha=args.adv_alpha,
        adv_delta=args.adv_delta,
        stability_weight=args.stability_weight,
        grad_norm_weight=args.grad_norm_weight,
        prox_lambda_bar_ratio=args.prox_lambda_bar_ratio,
    )
    path = selector.path(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
    desired = choose_target_checkpoint(path, args.k)
    selected_mask = desired.selected.cpu().numpy().astype(bool)
    selected_count = int(selected_mask.sum())
    if selected_count == 0:
        raise RuntimeError("Proposed method selected 0 features.")

    X_train_selected = split.X_train[:, selected_mask]
    X_val_selected = split.X_val[:, selected_mask]
    X_test_selected = split.X_test[:, selected_mask]

    refit_hidden_dim = (max(1, X_train_selected.shape[1] // 3),)
    refit = AdversarialLassoNetClassifier(
        M=10,
        hidden_dims=refit_hidden_dim,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=args.lasso_epochs,
        batch_size=args.batch_size,
        adv_rho=args.adv_rho,
        adv_alpha=args.adv_alpha,
        adv_delta=args.adv_delta,
        stability_weight=args.stability_weight,
        grad_norm_weight=args.grad_norm_weight,
        prox_lambda_bar_ratio=args.prox_lambda_bar_ratio,
    )
    path_sparse = refit.path(
        X_train_selected,
        split.y_train,
        X_val=X_val_selected,
        y_val=split.y_val,
        return_state_dicts=True,
    )[:1]
    val_score = float(eval_on_path(refit, path_sparse, X_val_selected, split.y_val)[0])
    test_score = float(eval_on_path(refit, path_sparse, X_test_selected, split.y_test)[0])
    return MethodResult(
        selected_count=selected_count,
        val_score=val_score,
        test_score=test_score,
        selected_lambda=float(desired.lambda_),
    )


def run_fista_net(
    split: DatasetSplit, seed: int, args: argparse.Namespace, device: torch.device
) -> MethodResult:
    hidden_dim = (max(8, split.data_dim // 3),)
    selector = FISTATabularClassifier(
        hidden_dims=hidden_dim,
        lambda_start=args.fista_lambda_start,
        path_multiplier=args.fista_path_multiplier,
        n_iters_init=args.fista_n_iters_init,
        n_iters_path=args.fista_n_iters_path,
        batch_size=args.fista_batch_size,
        lr=args.fista_lr,
        weight_decay=args.fista_weight_decay,
        device=str(device),
        verbose=args.fista_verbose,
        random_state=seed,
        torch_seed=seed,
    )
    path = selector.path(
        split.X_train,
        split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        return_state_dicts=True,
    )
    desired = choose_nonempty_fista_checkpoint(path, args.k)
    selected_mask = desired.selected.numpy().astype(bool)
    selected_count = int(selected_mask.sum())

    X_train_selected = split.X_train[:, selected_mask]
    X_val_selected = split.X_val[:, selected_mask]
    X_test_selected = split.X_test[:, selected_mask]

    refit_hidden = (max(4, selected_count // 2),)
    refit = FISTATabularClassifier(
        hidden_dims=refit_hidden,
        lambda_seq=[],
        n_iters_init=args.fista_n_iters_refit,
        n_iters_path=0,
        batch_size=args.fista_batch_size,
        lr=args.fista_lr,
        weight_decay=args.fista_weight_decay,
        device=str(device),
        verbose=args.fista_verbose,
        random_state=seed,
        torch_seed=seed,
    )
    refit.fit(X_train_selected, split.y_train, X_val=X_val_selected, y_val=split.y_val)
    val_score = refit.score(X_val_selected, split.y_val)
    test_score = refit.score(X_test_selected, split.y_test)
    return MethodResult(
        selected_count=selected_count,
        val_score=float(val_score),
        test_score=float(test_score),
        selected_lambda=float(desired.lambda_),
    )


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%"


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args=None) -> None:
    if args is None:
        args = parse_benchmark_args()
    device = resolve_device(args.device)
    datasets = resolve_benchmark_datasets(args.datasets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {[METHOD_LABELS[name] for name in METHOD_ORDER]}")

    run_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        print(f"\n===== Dataset: {dataset} =====")
        for run_idx in range(args.runs):
            seed = args.base_seed + run_idx
            set_seed(seed)
            split = load_split(dataset, seed)
            print(f"\n--- Seed {seed} ---")

            method_fns = {
                "vanilla_lassonet": run_vanilla_lassonet,
                "fista_net": run_fista_net,
                "deep_lasso": run_deep_lasso,
                "proposed_method": run_proposed_method,
            }

            for method_name in METHOD_ORDER:
                print(f"Running {METHOD_LABELS[method_name]} ...")
                result = method_fns[method_name](split, seed, args, device)
                run_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_display": DISPLAY_DATASETS.get(dataset, dataset),
                        "method": method_name,
                        "method_display": METHOD_LABELS[method_name],
                        "seed": seed,
                        "target_k": args.k,
                        "selected_count": result.selected_count,
                        "sparsity": 1.0 - (result.selected_count / split.data_dim),
                        "val_score": result.val_score,
                        "test_score": result.test_score,
                        "selected_lambda": result.selected_lambda,
                    }
                )
                print(
                    f"seed={seed} | method={METHOD_LABELS[method_name]} | "
                    f"selected={result.selected_count}/{split.data_dim} | "
                    f"val={result.val_score:.4f} | test={result.test_score:.4f}"
                )

    raw_csv_path = output_dir / f"{args.output_prefix}_raw_runs.csv"
    write_csv(
        raw_csv_path,
        [
            "dataset",
            "dataset_display",
            "method",
            "method_display",
            "seed",
            "target_k",
            "selected_count",
            "sparsity",
            "val_score",
            "test_score",
            "selected_lambda",
        ],
        run_rows,
    )

    summary_rows: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        dataset_runs = [row for row in run_rows if row["dataset"] == dataset]
        table_row: Dict[str, object] = {"Dataset": DISPLAY_DATASETS.get(dataset, dataset)}
        for method_name in METHOD_ORDER:
            method_runs = [row for row in dataset_runs if row["method"] == method_name]
            test_scores = np.array([float(row["test_score"]) for row in method_runs], dtype=float)
            selected_counts = np.array(
                [float(row["selected_count"]) for row in method_runs], dtype=float
            )
            val_scores = np.array([float(row["val_score"]) for row in method_runs], dtype=float)
            mean_test = float(test_scores.mean())
            std_test = float(test_scores.std(ddof=1)) if len(test_scores) > 1 else 0.0
            mean_selected = float(selected_counts.mean())
            std_selected = float(selected_counts.std(ddof=1)) if len(selected_counts) > 1 else 0.0
            mean_val = float(val_scores.mean())
            std_val = float(val_scores.std(ddof=1)) if len(val_scores) > 1 else 0.0

            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": DISPLAY_DATASETS.get(dataset, dataset),
                    "method": method_name,
                    "method_display": METHOD_LABELS[method_name],
                    "runs": len(method_runs),
                    "target_k": args.k,
                    "test_score_mean": mean_test,
                    "test_score_std": std_test,
                    "test_score_table2": format_percent(mean_test, std_test),
                    "val_score_mean": mean_val,
                    "val_score_std": std_val,
                    "selected_count_mean": mean_selected,
                    "selected_count_std": std_selected,
                }
            )
            table_row[METHOD_LABELS[method_name]] = format_percent(mean_test, std_test)
        table_rows.append(table_row)

    summary_csv_path = output_dir / f"{args.output_prefix}_summary.csv"
    write_csv(
        summary_csv_path,
        [
            "dataset",
            "dataset_display",
            "method",
            "method_display",
            "runs",
            "target_k",
            "test_score_mean",
            "test_score_std",
            "test_score_table2",
            "val_score_mean",
            "val_score_std",
            "selected_count_mean",
            "selected_count_std",
        ],
        summary_rows,
    )

    table2_csv_path = output_dir / f"{args.output_prefix}_table2.csv"
    write_csv(
        table2_csv_path,
        ["Dataset", *[METHOD_LABELS[name] for name in METHOD_ORDER]],
        table_rows,
    )

    print("\n===== Output =====")
    print(f"Raw runs CSV: {raw_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Table 2 CSV: {table2_csv_path}")


if __name__ == "__main__":
    main()
