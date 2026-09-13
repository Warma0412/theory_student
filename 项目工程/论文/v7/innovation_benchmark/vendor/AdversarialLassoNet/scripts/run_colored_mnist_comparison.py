import argparse
import itertools
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from adversarial_lassonet import LassoNetSAMAligned
from utils.colored_mnist_evaluation import Config, make_env_datasets, set_seed
from lassonet import LassoNetClassifier


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_adv: bool
    adv_rho: float = 0.0
    adv_alpha: float = 0.0
    adv_delta: float = 1e-12


@dataclass
class SingleRunResult:
    method: str
    seed: int
    selected_count: int
    sparsity: float
    id_accuracy: float
    ood_accuracy: float
    gap: float
    selected_mask: np.ndarray
    elapsed_seconds: float


def parse_colored_mnist_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vanilla LassoNet and adversarial LassoNet on Colored MNIST."
    )
    # Defaults target a moderate-cost main experiment rather than a smoke test.
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset-per-env", type=int, default=2000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--label-flip-prob", type=float, default=0.0)
    parser.add_argument("--train-env-probs", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--id-env-prob", type=float, default=0.2)
    parser.add_argument("--ood-env-prob", type=float, default=0.9)
    parser.add_argument("--intervention-env-prob", type=float, default=0.5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--k-tolerance", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lasso-epochs", type=int, default=20)
    parser.add_argument("--lasso-verbose", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--adv-rho", type=float, default=0.03)
    parser.add_argument("--adv-alpha", type=float, default=0.5)
    parser.add_argument("--adv-delta", type=float, default=1e-12)
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def parse_probabilities(text: str) -> Tuple[float, ...]:
    return tuple(float(token.strip()) for token in text.split(",") if token.strip())


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_arg


def build_eval_config(args: argparse.Namespace, device: str) -> Config:
    return Config(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=10,
        hidden_dim=args.hidden_dim,
        device=device,
        label_flip_prob=args.label_flip_prob,
        train_env_probs=parse_probabilities(args.train_env_probs),
        id_env_prob=args.id_env_prob,
        ood_env_prob=args.ood_env_prob,
        intervention_env_prob=args.intervention_env_prob,
        train_subset_per_env=args.train_subset_per_env,
        eval_subset=args.eval_subset,
        num_workers=0,
    )


def build_methods(args: argparse.Namespace) -> List[MethodConfig]:
    return [
        MethodConfig(name="lassonet", use_adv=False),
        MethodConfig(
            name="adv_lassonet",
            use_adv=True,
            adv_rho=args.adv_rho,
            adv_alpha=args.adv_alpha,
            adv_delta=args.adv_delta,
        ),
    ]


def flatten_envs(envs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    y_trues = []
    for env in envs:
        xs.append(env.x.view(env.x.size(0), -1).cpu().numpy())
        ys.append(env.y.cpu().numpy())
        y_trues.append(env.y_true.cpu().numpy())
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.int64),
        np.concatenate(y_trues, axis=0).astype(np.int64),
    )


def flatten_single_env(env) -> Tuple[np.ndarray, np.ndarray]:
    x = env.x.view(env.x.size(0), -1).cpu().numpy().astype(np.float32)
    y_true = env.y_true.cpu().numpy().astype(np.int64)
    return x, y_true


def jaccard_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / union)


def mean_pairwise_jaccard(masks: List[np.ndarray]) -> float:
    if len(masks) <= 1:
        return 1.0
    scores = [jaccard_score(a, b) for a, b in itertools.combinations(masks, 2)]
    return float(np.mean(scores))


def build_model(method: MethodConfig, args: argparse.Namespace, seed: int, device: str, input_dim: int):
    hidden_dims = (min(args.hidden_dim, max(1, input_dim // 2)),)
    common_kwargs = dict(
        M=10,
        hidden_dims=hidden_dims,
        verbose=args.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=(args.lasso_epochs, args.lasso_epochs),
        patience=(10, 10),
        batch_size=args.batch_size,
        backtrack=True,
    )
    if method.use_adv:
        return LassoNetSAMAligned(
            adv_rho=method.adv_rho,
            adv_alpha=method.adv_alpha,
            adv_delta=method.adv_delta,
            **common_kwargs,
        )
    return LassoNetClassifier(**common_kwargs)


def choose_target_save(path, k: int):
    return min(path, key=lambda item: abs(int(item.selected.sum().item()) - k))


def evaluate_true_accuracy(model, x: np.ndarray, y_true: np.ndarray) -> float:
    preds = model.predict(x)
    return float(np.mean(preds == y_true))


def train_one_run(method: MethodConfig, args: argparse.Namespace, seed: int, device: str) -> SingleRunResult:
    set_seed(seed)
    t0 = time.perf_counter()
    cfg = build_eval_config(args, device)
    train_envs, eval_envs = make_env_datasets(cfg)

    x_all, y_all, _ = flatten_envs(train_envs)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=args.val_size,
        random_state=seed,
        stratify=y_all,
    )

    model = build_model(method, args, seed, device, x_train.shape[1])
    path = model.path(
        x_train,
        y_train,
        X_val=x_val,
        y_val=y_val,
        return_state_dicts=True,
    )
    chosen = choose_target_save(path, args.k)
    selected_mask = chosen.selected.cpu().numpy().astype(bool)
    selected_count = int(selected_mask.sum())
    if selected_count == 0:
        raise RuntimeError(
            f"{method.name} seed={seed} selected 0 features. Reduce --k or adjust regularization."
        )
    if abs(selected_count - args.k) > args.k_tolerance:
        raise RuntimeError(
            f"{method.name} seed={seed} selected {selected_count} features, "
            f"which exceeds k tolerance around target k={args.k}."
        )

    x_train_sel = x_train[:, selected_mask]
    x_val_sel = x_val[:, selected_mask]
    final_model = build_model(method, args, seed, device, x_train_sel.shape[1])
    final_path = final_model.path(
        x_train_sel,
        y_train,
        X_val=x_val_sel,
        y_val=y_val,
        lambda_seq=[],
        return_state_dicts=False,
    )
    if not final_path:
        raise RuntimeError("Dense retraining on selected features did not produce a model.")

    x_id, y_id_true = flatten_single_env(eval_envs["id"])
    x_ood, y_ood_true = flatten_single_env(eval_envs["ood"])
    x_id_sel = x_id[:, selected_mask]
    x_ood_sel = x_ood[:, selected_mask]

    id_accuracy = evaluate_true_accuracy(final_model, x_id_sel, y_id_true)
    ood_accuracy = evaluate_true_accuracy(final_model, x_ood_sel, y_ood_true)
    elapsed = time.perf_counter() - t0

    return SingleRunResult(
        method=method.name,
        seed=seed,
        selected_count=selected_count,
        sparsity=1.0 - selected_count / x_all.shape[1],
        id_accuracy=id_accuracy,
        ood_accuracy=ood_accuracy,
        gap=id_accuracy - ood_accuracy,
        selected_mask=selected_mask,
        elapsed_seconds=float(elapsed),
    )


def summarize_results(results: List[SingleRunResult]) -> Dict[str, float]:
    id_acc = np.array([item.id_accuracy for item in results], dtype=np.float64)
    ood_acc = np.array([item.ood_accuracy for item in results], dtype=np.float64)
    gap = np.array([item.gap for item in results], dtype=np.float64)
    sparsity = np.array([item.sparsity for item in results], dtype=np.float64)
    selected = np.array([item.selected_count for item in results], dtype=np.float64)
    elapsed = np.array([item.elapsed_seconds for item in results], dtype=np.float64)
    jaccard = mean_pairwise_jaccard([item.selected_mask for item in results])

    def mean_std(values: np.ndarray, prefix: str) -> Dict[str, float]:
        ans = {f"{prefix}_mean": float(values.mean())}
        ans[f"{prefix}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return ans

    summary = {}
    summary.update(mean_std(id_acc, "id_accuracy"))
    summary.update(mean_std(ood_acc, "ood_accuracy"))
    summary.update(mean_std(gap, "gap"))
    summary.update(mean_std(sparsity, "sparsity"))
    summary.update(mean_std(selected, "selected_count"))
    summary.update(mean_std(elapsed, "elapsed_seconds"))
    summary["jaccard_mean"] = float(jaccard)
    summary["runs"] = len(results)
    return summary


def print_method_report(method: MethodConfig, results: List[SingleRunResult], summary: Dict[str, float]) -> None:
    print(f"\n===== {method.name} =====")
    for item in results:
        print(
            f"seed={item.seed} | selected={item.selected_count} | "
            f"sparsity={item.sparsity:.4f} | ID={item.id_accuracy:.4f} | "
            f"OOD={item.ood_accuracy:.4f} | gap={item.gap:.4f} | "
            f"time={item.elapsed_seconds:.2f}s"
        )

    print(
        "Summary | "
        f"ID={summary['id_accuracy_mean']:.4f} +- {summary['id_accuracy_std']:.4f} | "
        f"OOD={summary['ood_accuracy_mean']:.4f} +- {summary['ood_accuracy_std']:.4f} | "
        f"gap={summary['gap_mean']:.4f} +- {summary['gap_std']:.4f} | "
        f"sparsity={summary['sparsity_mean']:.4f} +- {summary['sparsity_std']:.4f} | "
        f"Jaccard={summary['jaccard_mean']:.4f}"
    )


def print_comparison_table(all_summaries: Dict[str, Dict[str, float]]) -> None:
    print("\n===== Final Comparison =====")
    print(
        f"{'method':<16} {'ID acc':>12} {'OOD acc':>12} {'gap':>12} "
        f"{'sparsity':>12} {'Jaccard':>12}"
    )
    for method_name, summary in all_summaries.items():
        print(
            f"{method_name:<16} "
            f"{summary['id_accuracy_mean']:.4f}+-{summary['id_accuracy_std']:.4f} "
            f"{summary['ood_accuracy_mean']:.4f}+-{summary['ood_accuracy_std']:.4f} "
            f"{summary['gap_mean']:.4f}+-{summary['gap_std']:.4f} "
            f"{summary['sparsity_mean']:.4f}+-{summary['sparsity_std']:.4f} "
            f"{summary['jaccard_mean']:.4f}"
        )


def main() -> None:
    args = parse_colored_mnist_args()
    device = resolve_device(args.device)
    methods = build_methods(args)

    print("Colored MNIST LassoNet comparison")
    print(f"device={device}")
    print(
        f"runs={args.runs}, base_seed={args.base_seed}, k={args.k}, k_tolerance={args.k_tolerance}, "
        f"lasso_epochs={args.lasso_epochs}, train_env_probs={parse_probabilities(args.train_env_probs)}, "
        f"id_env_prob={args.id_env_prob}, ood_env_prob={args.ood_env_prob}"
    )

    all_summaries = {}
    all_results = {}

    for method in methods:
        method_results = []
        for run_idx in range(args.runs):
            seed = args.base_seed + run_idx
            result = train_one_run(method, args, seed, device)
            method_results.append(result)
        summary = summarize_results(method_results)
        all_results[method.name] = method_results
        all_summaries[method.name] = summary
        print_method_report(method, method_results, summary)

    print_comparison_table(all_summaries)

    if args.save_json:
        output = {
            "config": vars(args),
            "device": device,
            "methods": {
                method_name: {
                    "summary": all_summaries[method_name],
                    "runs": [
                        {
                            "seed": item.seed,
                            "selected_count": item.selected_count,
                            "sparsity": item.sparsity,
                            "id_accuracy": item.id_accuracy,
                            "ood_accuracy": item.ood_accuracy,
                            "gap": item.gap,
                            "elapsed_seconds": item.elapsed_seconds,
                        }
                        for item in items
                    ],
                }
                for method_name, items in all_results.items()
            },
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
