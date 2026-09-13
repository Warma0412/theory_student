import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from adversarial_lassonet import (  # noqa: E402
    build_adv_arg_parser,
    canonical_dataset_name,
    namespace_to_adversarial_lassonet_config,
    resolve_datasets,
    resolve_device,
    save_sparse_checkpoint,
    train_adversarial_lassonet_once,
)
from utils.data_utils import load_dataset  # noqa: E402


def main() -> None:
    parser = build_adv_arg_parser(
        description="Run the enhanced adversarial-perturbation LassoNet training experiment."
    )
    args = parser.parse_args()
    config = namespace_to_adversarial_lassonet_config(args)
    device = resolve_device(args.device)

    if args.dataset:
        datasets = [canonical_dataset_name(args.dataset.strip())]
    else:
        datasets = resolve_datasets(args.datasets)

    summary = {}

    print(f"Device: {device}")
    print(f"Datasets: {datasets}")
    print(
        f"Config: runs={args.runs}, k={config.k}, lasso_epochs={config.lasso_epochs}, "
        f"adv_rho={config.adv_rho}, adv_alpha={config.adv_alpha}, "
        f"stability_weight={config.stability_weight}, grad_norm_weight={config.grad_norm_weight}, "
        f"prox_lambda_bar_ratio={config.prox_lambda_bar_ratio}"
    )

    for dataset in datasets:
        print(f"\n===== Dataset: {dataset} =====")
        run_selected = []
        run_val_accs = []
        run_test_accs = []

        for i in range(args.runs):
            seed = args.base_seed + i
            result = train_adversarial_lassonet_once(
                dataset=dataset,
                seed=seed,
                device=device,
                config=config,
                load_dataset_fn=load_dataset,
            )
            run_selected.append(result.selected_count)
            run_val_accs.append(result.val_acc)
            run_test_accs.append(result.test_acc)

            if args.save_pkl_dir:
                checkpoint_path = save_sparse_checkpoint(
                    output_dir=args.save_pkl_dir,
                    dataset=dataset,
                    seed=seed,
                    result=result,
                    config=config,
                )
                print(f"Saved checkpoint pkl: {checkpoint_path}")

            print(
                f"Run {i + 1}/{args.runs} seed={seed} | "
                f"selected={result.selected_count} | val={result.val_acc:.4f} | "
                f"test={result.test_acc:.4f}"
            )

        mean_sel = float(np.mean(run_selected))
        std_sel = float(np.std(run_selected, ddof=1)) if len(run_selected) > 1 else 0.0
        mean_val = float(np.mean(run_val_accs))
        std_val = float(np.std(run_val_accs, ddof=1)) if len(run_val_accs) > 1 else 0.0
        mean_test = float(np.mean(run_test_accs))
        std_test = float(np.std(run_test_accs, ddof=1)) if len(run_test_accs) > 1 else 0.0

        summary[dataset] = {
            "selected_mean": mean_sel,
            "selected_std": std_sel,
            "val_mean": mean_val,
            "val_std": std_val,
            "test_mean": mean_test,
            "test_std": std_test,
            "runs": args.runs,
        }

        print(
            f"Summary {dataset}: selected {mean_sel:.2f} +- {std_sel:.2f}, "
            f"val {mean_val:.4f} +- {std_val:.4f}, "
            f"test {mean_test:.4f} +- {std_test:.4f}"
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
