from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from adversarial_lassonet import LassoNetSAMAligned, canonical_dataset_name, set_seed
from utils.data_utils import load_dataset
from lassonet.utils import eval_on_path


def train_once(
    dataset: str,
    seed: int,
    alpha: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, int, float, float]:
    """Reuse the two-stage training pipeline from adversarial_lassonet.py
    and additionally return selected_mask for later Jaccard computation."""

    set_seed(seed)

    loaded = load_dataset(dataset)
    if loaded is None:
        raise ValueError(f"Invalid dataset: {dataset}")

    (X_train_valid, y_train_valid), (X_test, y_test) = loaded
    y_train_valid = y_train_valid.astype(np.int64)
    y_test = y_test.astype(np.int64)

    stratify = y_train_valid if len(np.unique(y_train_valid)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.125,
        random_state=seed,
        stratify=stratify,
    )

    data_dim = X_test.shape[1]
    hidden_dim = (max(1, data_dim // 3),)

    lasso_model = LassoNetSAMAligned(
        M=10,
        hidden_dims=hidden_dim,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=args.lasso_epochs,
        batch_size=args.batch_size,
        adv_rho=args.adv_rho,
        adv_alpha=alpha,
        adv_delta=args.adv_delta,
    )
    path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val)

    desired = min(path, key=lambda save: abs(int(save.selected.sum().item()) - args.k))
    selected_mask = desired.selected.cpu().numpy().astype(bool)
    selected_count = int(selected_mask.sum())
    if selected_count == 0:
        raise RuntimeError(
            f"{dataset}: selected 0 features; please reduce --k or adjust the sparsity strength."
        )

    X_train_sel = X_train[:, selected_mask]
    X_val_sel = X_val[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]

    sparse_hidden_dim = (max(1, X_train_sel.shape[1] // 3),)
    lasso_sparse = LassoNetSAMAligned(
        M=10,
        hidden_dims=sparse_hidden_dim,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=args.lasso_epochs,
        batch_size=args.batch_size,
        adv_rho=args.adv_rho,
        adv_alpha=alpha,
        adv_delta=args.adv_delta,
    )
    path_sparse = lasso_sparse.path(
        X_train_sel,
        y_train,
        X_val=X_val_sel,
        y_val=y_val,
        return_state_dicts=True,
    )[:1]

    val_acc = float(eval_on_path(lasso_sparse, path_sparse, X_val_sel, y_val)[0])
    test_acc = float(eval_on_path(lasso_sparse, path_sparse, X_test_sel, y_test)[0])
    return selected_mask, selected_count, val_acc, test_acc


def avg_pairwise_jaccard(masks: List[np.ndarray]) -> float:
    """Compute the average pairwise Jaccard similarity among selected_mask values across runs."""

    if len(masks) < 2:
        return 1.0
    sims: List[float] = []
    n = len(masks)
    for i in range(n):
        a = masks[i]
        for j in range(i + 1, n):
            b = masks[j]
            inter = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum())
            sims.append(inter / union if union > 0 else 0.0)
    return float(np.mean(sims))


def parse_sweep_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adv_alpha sweep on MICE and record val/test accuracy together with average Jaccard."
    )
    parser.add_argument("--dataset", type=str, default="MICE")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated training runs for each alpha.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50, help="Target number of selected features.")
    parser.add_argument("--lasso-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lasso-verbose", type=int, default=0)
    parser.add_argument("--adv-rho", type=float, default=0.1)
    parser.add_argument("--adv-delta", type=float, default=1e-12)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Candidate alpha values, separated by commas.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="alpha_sweep_results")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_sweep_args()
    device = resolve_device(args.device)
    dataset = canonical_dataset_name(args.dataset)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Alphas: {alphas}")
    print(f"Runs per alpha: {args.runs}")
    print(
        f"Config: k={args.k}, lasso_epochs={args.lasso_epochs}, "
        f"batch_size={args.batch_size}, adv_rho={args.adv_rho}"
    )

    summary: Dict[str, Dict[str, float]] = {}
    all_records: List[Dict[str, float]] = []

    for alpha in alphas:
        print(f"\n===== alpha = {alpha} =====")
        masks: List[np.ndarray] = []
        val_accs: List[float] = []
        test_accs: List[float] = []
        sel_counts: List[int] = []

        for i in range(args.runs):
            seed = args.base_seed + i
            mask, count, val_acc, test_acc = train_once(
                dataset, seed, alpha, args, device
            )
            masks.append(mask)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            sel_counts.append(count)
            print(
                f"  Run {i + 1}/{args.runs} seed={seed} | selected={count} | "
                f"val={val_acc:.4f} | test={test_acc:.4f}"
            )
            all_records.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "selected_count": count,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                }
            )

        jaccard = avg_pairwise_jaccard(masks)
        mean_val = float(np.mean(val_accs))
        std_val = float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0
        mean_test = float(np.mean(test_accs))
        std_test = float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0
        summary[f"{alpha}"] = {
            "alpha": alpha,
            "val_mean": mean_val,
            "val_std": std_val,
            "test_mean": mean_test,
            "test_std": std_test,
            "avg_jaccard": jaccard,
            "selected_mean": float(np.mean(sel_counts)),
            "runs": args.runs,
        }
        print(
            f"  -> val={mean_val:.4f} +- {std_val:.4f} | "
            f"test={mean_test:.4f} +- {std_test:.4f} | "
            f"avg_jaccard={jaccard:.4f}"
        )

    summary_path = output_dir / f"{dataset}_alpha_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "alphas": alphas,
                "config": {
                    "runs": args.runs,
                    "base_seed": args.base_seed,
                    "k": args.k,
                    "lasso_epochs": args.lasso_epochs,
                    "batch_size": args.batch_size,
                    "adv_rho": args.adv_rho,
                    "adv_delta": args.adv_delta,
                },
                "summary": summary,
                "records": all_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved summary -> {summary_path}")

    xs = alphas
    val_means = [summary[f"{a}"]["val_mean"] for a in alphas]
    val_stds = [summary[f"{a}"]["val_std"] for a in alphas]
    test_means = [summary[f"{a}"]["test_mean"] for a in alphas]
    test_stds = [summary[f"{a}"]["test_std"] for a in alphas]
    jaccs = [summary[f"{a}"]["avg_jaccard"] for a in alphas]

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.errorbar(
        xs, val_means, yerr=val_stds, marker="o", capsize=3, label="val_acc"
    )
    ax_acc.errorbar(
        xs, test_means, yerr=test_stds, marker="s", capsize=3, label="test_acc"
    )
    ax_acc.set_xlabel("adv_alpha")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{dataset}: accuracy vs adv_alpha (mean over {args.runs} runs)")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()
    fig_acc.tight_layout()
    acc_path = output_dir / f"{dataset}_acc_vs_alpha.png"
    fig_acc.savefig(acc_path, dpi=150)
    plt.close(fig_acc)
    print(f"Saved accuracy plot -> {acc_path}")

    fig_jac, ax_jac = plt.subplots(figsize=(8, 5))
    ax_jac.plot(xs, jaccs, marker="o", color="tab:green")
    ax_jac.set_xlabel("adv_alpha")
    ax_jac.set_ylabel("Avg pairwise Jaccard Index")
    ax_jac.set_title(
        f"{dataset}: feature-selection stability vs adv_alpha (over {args.runs} runs)"
    )
    ax_jac.grid(alpha=0.3)
    fig_jac.tight_layout()
    jac_path = output_dir / f"{dataset}_jaccard_vs_alpha.png"
    fig_jac.savefig(jac_path, dpi=150)
    plt.close(fig_jac)
    print(f"Saved Jaccard plot -> {jac_path}")

    fig_all, ax_left = plt.subplots(figsize=(9, 5))
    ax_left.errorbar(
        xs, val_means, yerr=val_stds, marker="o", capsize=3, color="tab:blue",
        label="val_acc",
    )
    ax_left.errorbar(
        xs, test_means, yerr=test_stds, marker="s", capsize=3, color="tab:orange",
        label="test_acc",
    )
    ax_left.set_xlabel("adv_alpha")
    ax_left.set_ylabel("Accuracy")
    ax_left.grid(alpha=0.3)
    ax_right = ax_left.twinx()
    ax_right.plot(xs, jaccs, marker="^", linestyle="--", color="tab:green",
                  label="avg_jaccard")
    ax_right.set_ylabel("Avg pairwise Jaccard Index")
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax_left.set_title(f"{dataset}: accuracy & Jaccard vs adv_alpha")
    fig_all.tight_layout()
    combo_path = output_dir / f"{dataset}_combined_vs_alpha.png"
    fig_all.savefig(combo_path, dpi=150)
    plt.close(fig_all)
    print(f"Saved combined plot -> {combo_path}")


if __name__ == "__main__":
    main()
