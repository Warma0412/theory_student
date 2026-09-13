import argparse
import csv
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from utils.data_utils import load_dataset
from lassonet import LassoNetClassifier
from lassonet.interfaces import HistoryItem


DATASET_ALIASES = {
    "coloredmnist": "ColoredMNIST",
    "colored-mnist": "ColoredMNIST",
    "colored_mnist": "ColoredMNIST",
    "ColoredMNIST": "ColoredMNIST",
}


def parse_ntk_args() -> argparse.Namespace:
    """Parse command-line arguments for the NTK spectrum analysis."""
    parser = argparse.ArgumentParser(
        description="Compare NTK spectrum statistics for LassoNet-style checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more model checkpoint paths. Pass two paths to compare vanilla and adv LassoNet on one figure.",
    )
    parser.add_argument(
        "--label",
        type=str,
        nargs="+",
        default=None,
        help="Optional legend labels matching --checkpoint order.",
    )
    parser.add_argument("--dataset", type=str, default="colored_mnist", help="Dataset name.")
    parser.add_argument("--subset_size", type=int, default=128, help="Evaluation subset size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Jacobian accumulation batch size.")
    parser.add_argument("--topk", type=int, default=20, help="Number of leading eigenvalues to save.")
    parser.add_argument("--output_dir", type=str, default="ntk_spectrum", help="Output directory.")
    parser.add_argument(
        "--logit_mode",
        type=str,
        default="true_class",
        choices=["true_class", "sum_logits"],
        help="Scalar output used for each sample. true_class is the default and most memory-safe mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for forward/backward passes.",
    )
    return parser.parse_args()


def canonical_dataset_name(name: str) -> str:
    """Return the project dataset name from a user-facing alias."""
    return DATASET_ALIASES.get(name, DATASET_ALIASES.get(name.lower(), name))


def resolve_device(device_arg: str) -> torch.device:
    """Resolve an auto/cpu/cuda device argument into a torch.device."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_subset(dataset: str, subset_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a small deterministic subset from the evaluation split.

    The existing project loader returns flattened Colored MNIST arrays, which is
    exactly the input format expected by LassoNet.
    """
    dataset = canonical_dataset_name(dataset)
    if dataset != "ColoredMNIST":
        raise ValueError("This script currently targets ColoredMNIST; use --dataset colored_mnist.")

    loaded = load_dataset(dataset)
    if loaded is None:
        raise ValueError(f"Invalid dataset: {dataset}")

    _, (x_eval, y_eval) = loaded
    n = min(int(subset_size), len(x_eval))
    if n <= 0:
        raise ValueError("--subset_size must be positive.")
    return x_eval[:n].astype(np.float32), y_eval[:n].astype(np.int64)


def read_checkpoint(path: Path):
    """Load a checkpoint with pickle for .pkl files and torch.load otherwise."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    return torch.load(path, map_location="cpu")


def choose_history_item(items: Iterable[HistoryItem]) -> HistoryItem:
    """Choose a HistoryItem from a path list, preferring the first saved state."""
    candidates = [item for item in items if isinstance(item, HistoryItem) and item.state_dict]
    if not candidates:
        raise ValueError("Checkpoint path list does not contain a HistoryItem with state_dict.")
    return candidates[0]


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    Extract a PyTorch state_dict from common checkpoint formats used in this repo.

    Supported formats include:
    - raw state_dict;
    - HistoryItem;
    - list/tuple of HistoryItem, such as run_lassonet_feature_selection.py's pickled path_sparse;
    - dict containing state_dict/model_state_dict/checkpoint/path.
    """
    if isinstance(checkpoint, HistoryItem):
        return checkpoint.state_dict

    if isinstance(checkpoint, (list, tuple)):
        return choose_history_item(checkpoint).state_dict

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and "skip.weight" in value:
                return value
            if isinstance(value, HistoryItem):
                return value.state_dict
        for key in ("path", "history", "path_sparse"):
            value = checkpoint.get(key)
            if isinstance(value, (list, tuple)):
                return choose_history_item(value).state_dict
        if "skip.weight" in checkpoint:
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected state_dict, HistoryItem, path list, "
        "or a dict containing one of those."
    )


def infer_hidden_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    """Infer LassoNet hidden_dims from layer weight shapes in a state_dict."""
    layer_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("layers.") and key.endswith(".weight")
        }
    )
    if not layer_indices:
        raise ValueError("Cannot infer hidden dimensions: no layers.*.weight keys found.")

    output_layer_index = layer_indices[-1]
    hidden_dims = [
        int(state_dict[f"layers.{idx}.weight"].shape[0])
        for idx in layer_indices
        if idx != output_layer_index
    ]
    return tuple(hidden_dims)


def extract_selected_mask(checkpoint) -> np.ndarray:
    """Extract an optional feature mask saved with sparse checkpoints."""
    if isinstance(checkpoint, dict):
        mask = checkpoint.get("selected_mask")
        if mask is not None:
            return np.asarray(mask, dtype=bool)
    return None


def load_lassonet_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, np.ndarray]:
    """
    Load a LassoNet torch module from a checkpoint.

    Both vanilla LassoNetClassifier and adv_before_ablation.LassoNetSAMAligned
    use the same underlying lassonet.model.LassoNet architecture, so the
    spectrum is computed on that shared PyTorch module.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    selected_mask = extract_selected_mask(checkpoint)
    hidden_dims = infer_hidden_dims(state_dict)

    model_wrapper = LassoNetClassifier(hidden_dims=hidden_dims, device=str(device), verbose=0)
    model_wrapper.load(state_dict)
    model = model_wrapper.model.to(device)
    model.eval()
    return model, selected_mask


def align_eval_features(x_eval: np.ndarray, model: nn.Module, selected_mask: np.ndarray) -> np.ndarray:
    """Slice evaluation features when a sparse checkpoint includes its original mask."""
    expected_dim = int(model.skip.weight.shape[1])
    if x_eval.shape[1] == expected_dim:
        return x_eval

    if selected_mask is not None:
        if selected_mask.shape[0] != x_eval.shape[1]:
            raise ValueError(
                "selected_mask length does not match evaluation feature dimension: "
                f"{selected_mask.shape[0]} vs {x_eval.shape[1]}"
            )
        x_selected = x_eval[:, selected_mask]
        if x_selected.shape[1] != expected_dim:
            raise ValueError(
                "Selected feature count does not match checkpoint input dimension: "
                f"{x_selected.shape[1]} vs {expected_dim}"
            )
        return x_selected.astype(np.float32)

    raise ValueError(
        "Checkpoint input dimension does not match evaluation data and no selected_mask "
        "was found in the checkpoint. Save sparse checkpoints with selected_mask, or use "
        "a checkpoint trained on the full feature space."
    )


def trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """Return trainable parameters in a stable order."""
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters.")
    return params


def flatten_grads(grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten a tuple of gradient tensors into one 1D vector on CPU."""
    return torch.cat([grad.detach().reshape(-1).cpu() for grad in grads])


def scalar_output_for_sample(logits: torch.Tensor, label: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Select the scalar output whose parameter gradient defines one Jacobian row.

    true_class uses the logit corresponding to the sample's ground-truth label.
    sum_logits is a simple fallback that still yields one scalar per sample.
    """
    if mode == "true_class":
        return logits[label.long()]
    if mode == "sum_logits":
        return logits.sum()
    raise ValueError(f"Unsupported logit mode: {mode}")


def compute_jacobian(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    logit_mode: str,
) -> torch.Tensor:
    """
    Compute per-sample Jacobian rows in mini-batches.

    The returned tensor is stored on CPU to keep GPU memory bounded by one
    mini-batch and one gradient vector at a time.
    """
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    params = trainable_parameters(model)
    rows = []

    for start in range(0, len(x_eval), batch_size):
        stop = min(start + batch_size, len(x_eval))
        xb = torch.as_tensor(x_eval[start:stop], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y_eval[start:stop], dtype=torch.long, device=device)

        for i in range(xb.shape[0]):
            model.zero_grad(set_to_none=True)
            logits = model(xb[i : i + 1]).squeeze(0)
            scalar = scalar_output_for_sample(logits, yb[i], logit_mode)
            grads = torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)
            rows.append(flatten_grads(grads))

    return torch.stack(rows, dim=0).to(torch.float64)


def compute_ntk_metrics(theta: torch.Tensor, topk: int) -> Dict[str, object]:
    """Compute eigenvalue-based NTK spectrum metrics."""
    theta = 0.5 * (theta + theta.T)
    eigvals = torch.linalg.eigvalsh(theta).flip(0).clamp_min(0.0).cpu().numpy()
    trace = float(eigvals.sum())
    fro_sq = float(np.square(eigvals).sum())

    if trace <= 0.0:
        effective_rank = 0.0
        lambda1_ratio = 0.0
        top10_energy_ratio = 0.0
        cumulative_energy = np.zeros_like(eigvals)
    else:
        effective_rank = float((trace * trace) / fro_sq) if fro_sq > 0 else 0.0
        lambda1_ratio = float(eigvals[0] / trace) if eigvals.size else 0.0
        top10_energy_ratio = float(eigvals[:10].sum() / trace)
        cumulative_energy = np.cumsum(eigvals) / trace

    return {
        "eigenvalues": eigvals.tolist(),
        "top_eigenvalues": eigvals[:topk].tolist(),
        "trace": trace,
        "frobenius_norm": float(np.sqrt(fro_sq)),
        "effective_rank": effective_rank,
        "lambda1_trace_ratio": lambda1_ratio,
        "top10_energy_ratio": top10_energy_ratio,
        "cumulative_energy": cumulative_energy.tolist(),
    }


def plot_spectrum(eigenvalues: List[float], path: Path, title: str) -> None:
    """Save the eigenvalue decay curve."""
    plt.figure(figsize=(7, 4.5))
    xs = np.arange(1, len(eigenvalues) + 1)
    plt.plot(xs, eigenvalues, marker="o", linewidth=1.5, markersize=3)
    plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_spectrum_comparison(results: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    """Save one eigenvalue decay figure with multiple checkpoints overlaid."""
    plt.figure(figsize=(7, 4.5))
    for result in results:
        eigenvalues = result["eigenvalues"]
        xs = np.arange(1, len(eigenvalues) + 1)
        plt.plot(xs, eigenvalues, marker="o", linewidth=1.5, markersize=3, label=result["label"])
    plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cumulative_energy(cumulative_energy: List[float], path: Path, title: str) -> None:
    """Save the cumulative spectral energy curve."""
    plt.figure(figsize=(7, 4.5))
    xs = np.arange(1, len(cumulative_energy) + 1)
    plt.plot(xs, cumulative_energy, marker="o", linewidth=1.5, markersize=3)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Cumulative energy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cumulative_energy_comparison(results: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    """Save one cumulative energy figure with multiple checkpoints overlaid."""
    plt.figure(figsize=(7, 4.5))
    for result in results:
        cumulative_energy = result["cumulative_energy"]
        xs = np.arange(1, len(cumulative_energy) + 1)
        plt.plot(xs, cumulative_energy, marker="o", linewidth=1.5, markersize=3, label=result["label"])
    plt.ylim(0.0, 1.02)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Cumulative energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def append_csv_row(path: Path, row: Dict[str, object], topk: int) -> None:
    """Append one result row to a CSV file, creating the header if needed."""
    eigen_cols = {f"lambda_{i + 1}": row["top_eigenvalues"][i] if i < len(row["top_eigenvalues"]) else "" for i in range(topk)}
    csv_row = {
        "timestamp": row["timestamp"],
        "label": row.get("label", ""),
        "checkpoint": row["checkpoint"],
        "dataset": row["dataset"],
        "subset_size": row["subset_size"],
        "batch_size": row["batch_size"],
        "logit_mode": row["logit_mode"],
        "trace": row["trace"],
        "frobenius_norm": row["frobenius_norm"],
        "effective_rank": row["effective_rank"],
        "lambda1_trace_ratio": row["lambda1_trace_ratio"],
        "top10_energy_ratio": row["top10_energy_ratio"],
        **eigen_cols,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)


def safe_stem(path: Path) -> str:
    """Create a filesystem-friendly name from a checkpoint path."""
    return path.stem.replace(" ", "_")


def normalize_labels(checkpoints: Sequence[Path], labels: Optional[Sequence[str]] = None) -> List[str]:
    """Return plot labels aligned with the checkpoint list."""
    if labels is None:
        return [safe_stem(path) for path in checkpoints]
    if len(labels) != len(checkpoints):
        raise ValueError("--label count must match --checkpoint count.")
    return list(labels)


def analyze_checkpoint(
    checkpoint_path: Path,
    label: str,
    *,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
    topk: int,
    logit_mode: str,
    dataset: str,
    device: torch.device,
) -> Dict[str, object]:
    """Compute NTK metrics for one checkpoint and package the result."""
    model, selected_mask = load_lassonet_model(checkpoint_path, device)
    x_eval_aligned = align_eval_features(x_eval, model, selected_mask)

    jacobian = compute_jacobian(
        model,
        x_eval_aligned,
        y_eval,
        batch_size=batch_size,
        device=device,
        logit_mode=logit_mode,
    )
    theta = jacobian @ jacobian.T
    metrics = compute_ntk_metrics(theta, topk)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "timestamp": timestamp,
        "label": label,
        "checkpoint": str(checkpoint_path),
        "dataset": canonical_dataset_name(dataset),
        "subset_size": int(len(x_eval_aligned)),
        "batch_size": int(batch_size),
        "topk": int(topk),
        "logit_mode": logit_mode,
        "device": str(device),
        "jacobian_shape": list(jacobian.shape),
        "theta_shape": list(theta.shape),
        **metrics,
    }


def main() -> None:
    """Run NTK spectrum analysis and save JSON/CSV/PNG outputs."""
    args = parse_ntk_args()
    device = resolve_device(args.device)
    checkpoint_paths = [Path(path) for path in args.checkpoint]
    labels = normalize_labels(checkpoint_paths, args.label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_eval, y_eval = load_eval_subset(args.dataset, args.subset_size)
    csv_path = output_dir / "ntk_spectrum_summary.csv"
    compare_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: List[Dict[str, object]] = []

    for checkpoint_path, label in zip(checkpoint_paths, labels):
        result = analyze_checkpoint(
            checkpoint_path,
            label,
            x_eval=x_eval,
            y_eval=y_eval,
            batch_size=args.batch_size,
            topk=args.topk,
            logit_mode=args.logit_mode,
            dataset=args.dataset,
            device=device,
        )
        stem = safe_stem(checkpoint_path)
        prefix = f"{stem}_ntk_{result['timestamp']}"
        json_path = output_dir / f"{prefix}.json"
        decay_png = output_dir / f"{prefix}_eigen_decay.png"
        energy_png = output_dir / f"{prefix}_cumulative_energy.png"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        append_csv_row(csv_path, result, args.topk)
        plot_spectrum(result["eigenvalues"], decay_png, f"NTK eigenvalue decay: {label}")
        plot_cumulative_energy(result["cumulative_energy"], energy_png, f"NTK cumulative energy: {label}")

        results.append(result)
        print(f"Saved JSON: {json_path}")
        print(f"Saved eigenvalue decay PNG: {decay_png}")
        print(f"Saved cumulative energy PNG: {energy_png}")
        print(
            f"[{label}] Metrics: "
            f"effective_rank={result['effective_rank']:.6f}, "
            f"lambda1/trace={result['lambda1_trace_ratio']:.6f}, "
            f"top10_energy={result['top10_energy_ratio']:.6f}"
        )

    print(f"Appended CSV: {csv_path}")

    if len(results) > 1:
        joined_labels = "_vs_".join(label.replace(" ", "_") for label in labels)
        compare_decay_png = output_dir / f"{joined_labels}_ntk_compare_{compare_timestamp}_eigen_decay.png"
        compare_energy_png = output_dir / f"{joined_labels}_ntk_compare_{compare_timestamp}_cumulative_energy.png"
        compare_json = output_dir / f"{joined_labels}_ntk_compare_{compare_timestamp}.json"

        plot_spectrum_comparison(results, compare_decay_png, "NTK eigenvalue decay comparison")
        plot_cumulative_energy_comparison(results, compare_energy_png, "NTK cumulative energy comparison")

        with compare_json.open("w", encoding="utf-8") as f:
            json.dump({"timestamp": compare_timestamp, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"Saved comparison JSON: {compare_json}")
        print(f"Saved comparison eigenvalue decay PNG: {compare_decay_png}")
        print(f"Saved comparison cumulative energy PNG: {compare_energy_png}")


if __name__ == "__main__":
    main()
