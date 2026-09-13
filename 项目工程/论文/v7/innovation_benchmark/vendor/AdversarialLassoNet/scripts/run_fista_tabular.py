import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths

add_project_src_paths(ROOT)

from adversarial_lassonet.paths import OUTPUTS_ROOT, ensure_directory

from models.fista_tabular import FISTATabularClassifier
from utils.data_utils import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ALIAS = {
    "mnist": "MNIST",
    "minst": "MNIST",
    "coloredmnist": "ColoredMNIST",
    "colored-mnist": "ColoredMNIST",
    "colored_mnist": "ColoredMNIST",
    "mnist-fashion": "MNIST-Fashion",
    "mice": "MICE",
    "coil": "COIL",
    "isolet": "ISOLET",
    "activity": "Activity",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FISTA-inspired tabular benchmark runner aligned with scripts/run_lassonet_feature_selection.py"
    )
    parser.add_argument("--dataset", type=str, default="MICE")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-iters-init", type=int, default=200)
    parser.add_argument("--n-iters-path", type=int, default=100)
    parser.add_argument("--n-iters-refit", type=int, default=300)
    parser.add_argument("--lambda-start", type=float, default=1e-4)
    parser.add_argument("--path-multiplier", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_ROOT / "exp1_main_benchmark" / "fista_tabular"),
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(dataset: str) -> str:
    return DATASET_ALIAS.get(dataset.lower(), dataset)


def choose_nonempty_checkpoint(path, target_k: int):
    nonempty = [save for save in path if int(save.selected.sum().item()) > 0]
    if not nonempty:
        raise RuntimeError(
            "FISTA path selected 0 features for every checkpoint. "
            "Try a smaller --lambda-start or a smaller --path-multiplier."
        )
    return min(nonempty, key=lambda save: abs(int(save.selected.sum().item()) - target_k))


def run_once(args, dataset: str, seed: int):
    set_seed(seed)

    loaded = load_dataset(dataset)
    valid = ", ".join(
        ["MNIST", "ColoredMNIST", "MNIST-Fashion", "MICE", "COIL", "ISOLET", "Activity"]
    )
    if loaded is None:
        raise ValueError(f"Invalid dataset '{dataset}'. Valid options: {valid}")

    (X_train_valid, y_train_valid), (X_test, y_test) = loaded
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, y_train_valid, test_size=0.125, random_state=seed
    )

    data_dim = X_test.shape[1]
    hidden_dim = (max(8, data_dim // 3),)

    selector = FISTATabularClassifier(
        hidden_dims=hidden_dim,
        lambda_start=args.lambda_start,
        path_multiplier=args.path_multiplier,
        n_iters_init=args.n_iters_init,
        n_iters_path=args.n_iters_path,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        verbose=args.verbose,
        random_state=seed,
        torch_seed=seed,
    )
    path = selector.path(X_train, y_train, X_val=X_val, y_val=y_val, return_state_dicts=True)

    desired_save = choose_nonempty_checkpoint(path, args.k)
    selected_features = desired_save.selected.numpy().astype(bool)
    selected_count = int(selected_features.sum())

    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    refit_hidden = (max(4, selected_count // 2),)
    refit = FISTATabularClassifier(
        hidden_dims=refit_hidden,
        lambda_seq=[],
        n_iters_init=args.n_iters_refit,
        n_iters_path=0,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        verbose=args.verbose,
        random_state=seed,
        torch_seed=seed,
    )
    refit.fit(X_train_selected, y_train, X_val=X_val_selected, y_val=y_val)
    score = refit.score(X_test_selected, y_test)

    return selected_count, score, path


def main():
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    output_dir = ensure_directory(Path(args.output_dir))

    selected_counts = []
    test_scores = []

    for run_idx in range(args.n_runs):
        current_seed = args.base_seed + run_idx
        print(f"\n===== Run {run_idx + 1}/{args.n_runs} (seed={current_seed}) =====")
        selected_count, test_score, path = run_once(args, dataset, current_seed)
        selected_counts.append(selected_count)
        test_scores.append(test_score)
        print("Number of selected features:", selected_count)
        print("Test accuracy:", test_score)

        with (output_dir / f"{dataset}_fista_path_seed{current_seed}.pkl").open("wb") as f:
            pickle.dump(path, f)

    print("\n===== Summary =====")
    if len(selected_counts) > 1:
        count_std = np.std(selected_counts, ddof=1)
        score_std = np.std(test_scores, ddof=1)
    else:
        count_std = 0.0
        score_std = 0.0
    print(
        f"Selected features mean +- std: {np.mean(selected_counts):.2f} +- {count_std:.2f}"
    )
    print(f"Test accuracy mean +- std: {np.mean(test_scores):.6f} +- {score_std:.6f}")


if __name__ == "__main__":
    main()
