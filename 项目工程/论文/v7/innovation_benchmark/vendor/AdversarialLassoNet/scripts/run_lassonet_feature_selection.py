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

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from utils.data_utils import load_dataset
from lassonet import LassoNetClassifier
from lassonet.utils import eval_on_path

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
K = 50  # Number of features to select
n_epochs = 1000
dataset = "colored_mnist"
n_runs = 1
base_seed = 42
save_pkl_dir = Path(".")

dataset_alias = {
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
dataset = dataset_alias.get(dataset.lower(), dataset)
valid = ", ".join(
    ["MNIST", "ColoredMNIST", "MNIST-Fashion", "MICE", "COIL", "ISOLET", "Activity"]
)


def save_sparse_checkpoint(dataset, seed, selected_features, path_sparse):
    """Save a sparse checkpoint with selected_mask for use by the NTK spectrum analysis script."""
    save_pkl_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "method": "lassonet",
        "selected_mask": selected_features.cpu().numpy().astype(bool),
        "selected_count": int(selected_features.sum().item()),
        "path_sparse": path_sparse,
    }
    out_path = save_pkl_dir / f"{dataset}_lassonet_path_seed{seed}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return out_path


def run_single_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loaded = load_dataset(dataset)
    if loaded is None:
        raise ValueError(f"Invalid dataset '{dataset}'. Valid options: {valid}")

    (X_train_valid, y_train_valid), (X_test, y_test) = loaded
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, y_train_valid, test_size=0.125, random_state=seed
    )

    data_dim = X_test.shape[1]
    hidden_dim = (data_dim // 3,)

    lasso_model = LassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
        batch_size=batch_size,
    )
    path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val)

    desired_save = min(path, key=lambda save: abs(int(save.selected.sum().item()) - K))
    selected_features = desired_save.selected
    selected_count = int(selected_features.sum().item())

    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    lasso_sparse = LassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
    )
    path_sparse = lasso_sparse.path(
        X_train_selected,
        y_train,
        X_val=X_val_selected,
        y_val=y_val,
        return_state_dicts=True,
    )[:1]

    score = eval_on_path(lasso_sparse, path_sparse, X_test_selected, y_test)[0]
    return selected_count, float(score), path_sparse, selected_features


selected_counts = []
test_scores = []

for run_idx in range(n_runs):
    current_seed = base_seed + run_idx
    print(f"\n===== Run {run_idx + 1}/{n_runs} (seed={current_seed}) =====")
    selected_count, test_score, path_sparse, selected_features = run_single_experiment(
        current_seed
    )
    selected_counts.append(selected_count)
    test_scores.append(test_score)
    print("Number of selected features:", selected_count)
    print("Test accuracy:", test_score)

    with open(f"{dataset}_path_seed{current_seed}.pkl", "wb") as f:
        pickle.dump(path_sparse, f)

    rich_checkpoint_path = save_sparse_checkpoint(
        dataset=dataset,
        seed=current_seed,
        selected_features=selected_features,
        path_sparse=path_sparse,
    )
    print("Saved NTK-ready checkpoint:", rich_checkpoint_path)

print("\n===== Summary =====")
print(
    f"Selected features mean +- std: "
    f"{np.mean(selected_counts):.2f} +- {np.std(selected_counts, ddof=1):.2f}"
)
print(
    f"Test accuracy mean +- std: "
    f"{np.mean(test_scores):.6f} +- {np.std(test_scores, ddof=1):.6f}"
)
