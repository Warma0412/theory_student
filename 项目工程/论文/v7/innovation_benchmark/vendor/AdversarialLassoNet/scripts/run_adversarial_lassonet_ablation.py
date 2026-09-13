import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  

add_project_src_paths(ROOT)

from adversarial_lassonet import (  
    AdversarialLassoNetConfig,
    build_adv_arg_parser,
    canonical_dataset_name,
    namespace_to_adversarial_lassonet_config,
    resolve_datasets,
    resolve_device,
    train_adversarial_lassonet_once,
)
from utils.data_utils import load_dataset  


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    adv_alpha: float
    stability_weight: float
    grad_norm_weight: float
    prox_lambda_bar_ratio: float


@dataclass
class RunResult:
    ablation: str
    data_dim: int
    selected_count: int
    sparsity: float
    val_acc: float
    test_acc: float
    elapsed_seconds: float
    selected_mask: np.ndarray


def build_ablation_configs(args) -> Dict[str, AblationConfig]:
    return {
        "A1": AblationConfig(
            name="A1",
            description="LassoNet",
            adv_alpha=0.0,
            stability_weight=0.0,
            grad_norm_weight=0.0,
            prox_lambda_bar_ratio=0.0,
        ),
        "A2": AblationConfig(
            name="A2",
            description="LassoNet + stability loss only",
            adv_alpha=0.0,
            stability_weight=args.stability_weight,
            grad_norm_weight=0.0,
            prox_lambda_bar_ratio=0.0,
        ),
        "A3": AblationConfig(
            name="A3",
            description="LassoNet + adv perturbation without hierarchical prox change",
            adv_alpha=args.adv_alpha,
            stability_weight=0.0,
            grad_norm_weight=0.0,
            prox_lambda_bar_ratio=0.0,
        ),
        "A4": AblationConfig(
            name="A4",
            description="LassoNet + grad norm penalty",
            adv_alpha=0.0,
            stability_weight=0.0,
            grad_norm_weight=args.grad_norm_weight,
            prox_lambda_bar_ratio=0.0,
        ),
        "A5": AblationConfig(
            name="A5",
            description="Full model",
            adv_alpha=args.adv_alpha,
            stability_weight=args.stability_weight,
            grad_norm_weight=args.grad_norm_weight,
            prox_lambda_bar_ratio=args.prox_lambda_bar_ratio,
        ),
    }


def resolve_ablation_sequence(args) -> List[AblationConfig]:
    configs = build_ablation_configs(args)
    if args.ablation.upper() == "ALL":
        return [configs[key] for key in ("A1", "A2", "A3", "A4", "A5")]

    names = [token.strip().upper() for token in args.ablation.split(",") if token.strip()]
    unknown = [name for name in names if name not in configs]
    if unknown:
        raise ValueError(f"Unknown ablation(s): {unknown}")
    return [configs[name] for name in names]


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


def build_config_for_ablation(
    args, ablation: AblationConfig
) -> AdversarialLassoNetConfig:
    base_config = namespace_to_adversarial_lassonet_config(args)
    return AdversarialLassoNetConfig(
        k=base_config.k,
        lasso_epochs=base_config.lasso_epochs,
        batch_size=base_config.batch_size,
        lasso_verbose=base_config.lasso_verbose,
        M=base_config.M,
        adv_rho=base_config.adv_rho,
        adv_alpha=ablation.adv_alpha,
        adv_delta=base_config.adv_delta,
        stability_weight=ablation.stability_weight,
        grad_norm_weight=ablation.grad_norm_weight,
        prox_lambda_bar_ratio=ablation.prox_lambda_bar_ratio,
    )


def train_one_dataset_once(dataset: str, seed: int, args, device, ablation: AblationConfig) -> RunResult:
    loaded = load_dataset(dataset)
    if loaded is None:
        raise ValueError(f"Invalid dataset: {dataset}")

    (_, _), (X_test, _) = loaded
    data_dim = X_test.shape[1]

    config = build_config_for_ablation(args, ablation)
    t0 = time.perf_counter()
    result = train_adversarial_lassonet_once(
        dataset=dataset,
        seed=seed,
        device=device,
        config=config,
        load_dataset_fn=load_dataset,
    )
    elapsed_seconds = float(time.perf_counter() - t0)
    sparsity = 1.0 - (result.selected_count / data_dim)

    return RunResult(
        ablation=ablation.name,
        data_dim=data_dim,
        selected_count=result.selected_count,
        sparsity=sparsity,
        val_acc=result.val_acc,
        test_acc=result.test_acc,
        elapsed_seconds=elapsed_seconds,
        selected_mask=result.selected_mask,
    )


def parse_ablation_args():
    parser = build_adv_arg_parser(
        description="LassoNet ablation runner with adversarial perturbation and extra regularizers.",
        include_save_pkl_dir=False,
    )
    parser.add_argument("--ablation", type=str, default="all", help='A1,A2,... or "all".')
    return parser.parse_args()


def main() -> None:
    args = parse_ablation_args()
    device = resolve_device(args.device)
    ablations = resolve_ablation_sequence(args)

    if args.dataset:
        datasets = [canonical_dataset_name(args.dataset.strip())]
    else:
        datasets = resolve_datasets(args.datasets)

    summary: Dict[str, Dict[str, Dict[str, object]]] = {}

    print(f"Device: {device}")
    print(f"Datasets: {datasets}")
    print(
        f"Config: runs={args.runs}, k={args.k}, lasso_epochs={args.lasso_epochs}, "
        f"adv_rho={args.adv_rho}, adv_alpha={args.adv_alpha}, "
        f"stability_weight={args.stability_weight}, grad_norm_weight={args.grad_norm_weight}, "
        f"prox_lambda_bar_ratio={args.prox_lambda_bar_ratio}"
    )
    print("Ablations:", [f"{item.name}:{item.description}" for item in ablations])

    for ds in datasets:
        print(f"\n===== Dataset: {ds} =====")
        summary[ds] = {}

        for ablation in ablations:
            print(f"\n--- {ablation.name}: {ablation.description} ---")
            run_selected = []
            run_sparsity = []
            run_val_accs = []
            run_test_accs = []
            run_times = []
            run_masks = []

            for i in range(args.runs):
                seed = args.base_seed + i
                result = train_one_dataset_once(ds, seed, args, device, ablation)
                run_selected.append(result.selected_count)
                run_sparsity.append(result.sparsity)
                run_val_accs.append(result.val_acc)
                run_test_accs.append(result.test_acc)
                run_times.append(result.elapsed_seconds)
                run_masks.append(result.selected_mask)
                print(
                    f"Run {i + 1}/{args.runs} seed={seed} | "
                    f"selected={result.selected_count}/{result.data_dim} | "
                    f"sparsity={result.sparsity:.4f} | val={result.val_acc:.4f} | "
                    f"test={result.test_acc:.4f} | time={result.elapsed_seconds:.2f}s"
                )

            mean_sel = float(np.mean(run_selected))
            std_sel = float(np.std(run_selected, ddof=1)) if len(run_selected) > 1 else 0.0
            mean_sparsity = float(np.mean(run_sparsity))
            std_sparsity = float(np.std(run_sparsity, ddof=1)) if len(run_sparsity) > 1 else 0.0
            mean_val = float(np.mean(run_val_accs))
            std_val = float(np.std(run_val_accs, ddof=1)) if len(run_val_accs) > 1 else 0.0
            mean_test = float(np.mean(run_test_accs))
            std_test = float(np.std(run_test_accs, ddof=1)) if len(run_test_accs) > 1 else 0.0
            mean_time = float(np.mean(run_times))
            std_time = float(np.std(run_times, ddof=1)) if len(run_times) > 1 else 0.0
            mean_jaccard = mean_pairwise_jaccard(run_masks)

            summary[ds][ablation.name] = {
                "description": ablation.description,
                "selected_mean": mean_sel,
                "selected_std": std_sel,
                "sparsity_mean": mean_sparsity,
                "sparsity_std": std_sparsity,
                "val_mean": mean_val,
                "val_std": std_val,
                "test_mean": mean_test,
                "test_std": std_test,
                "jaccard_mean": mean_jaccard,
                "time_mean_seconds": mean_time,
                "time_std_seconds": std_time,
                "runs": args.runs,
            }

            print(
                f"Summary {ablation.name}: selected {mean_sel:.2f} +- {std_sel:.2f}, "
                f"sparsity {mean_sparsity:.4f} +- {std_sparsity:.4f}, "
                f"val {mean_val:.4f} +- {std_val:.4f}, "
                f"test {mean_test:.4f} +- {std_test:.4f}, "
                f"jaccard {mean_jaccard:.4f}, "
                f"time {mean_time:.2f}s +- {std_time:.2f}s"
            )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
