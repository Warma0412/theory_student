import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from adversarial_lassonet import LassoNetSAMAligned
from utils.data_utils import load_dataset
from lassonet import LassoNetClassifier
from lassonet.interfaces import HistoryItem


DATASET_ALIASES = {
    "coloredmnist": "ColoredMNIST",
    "colored-mnist": "ColoredMNIST",
    "colored_mnist": "ColoredMNIST",
    "ColoredMNIST": "ColoredMNIST",
}


@dataclass
class CheckpointMeta:
    method: str
    config: Dict[str, float]
    selected_mask: Optional[np.ndarray]


def parse_hessian_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate Hessian spectrum statistics for LassoNet checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more model checkpoint paths. Pass multiple checkpoints for one-figure comparison.",
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
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size for HVP accumulation.")
    parser.add_argument("--power_iters", type=int, default=30, help="Power iteration steps.")
    parser.add_argument(
        "--hutchinson_samples",
        type=int,
        default=20,
        help="Number of Hutchinson random vectors.",
    )
    parser.add_argument("--output_dir", type=str, default="hessian_spectrum", help="Output directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Prefer GPU when available, otherwise fall back automatically.",
    )
    parser.add_argument(
        "--lanczos_steps",
        type=int,
        default=20,
        help="Lanczos steps for eigenvalue curve and density approximation.",
    )
    return parser.parse_args()


def canonical_dataset_name(name: str) -> str:
    """Normalize dataset aliases to the project dataset name."""
    return DATASET_ALIASES.get(name, DATASET_ALIASES.get(name.lower(), name))


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the user device choice."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_subset(dataset: str, subset_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a deterministic subset from the evaluation split."""
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


def build_eval_loader(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Create an evaluation dataloader from a numpy subset."""
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    dataset = TensorDataset(
        torch.from_numpy(x_eval).float(),
        torch.from_numpy(y_eval).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def read_checkpoint(path: Path):
    """Load a checkpoint with pickle for .pkl files and torch.load otherwise."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    return torch.load(path, map_location="cpu")


def choose_history_item(items: Iterable[HistoryItem]) -> HistoryItem:
    """Choose a HistoryItem from a saved path list."""
    candidates = [item for item in items if isinstance(item, HistoryItem) and item.state_dict]
    if not candidates:
        raise ValueError("Checkpoint path list does not contain a HistoryItem with state_dict.")
    return candidates[0]


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict from common checkpoint formats used in this repo."""
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


def extract_checkpoint_meta(checkpoint) -> CheckpointMeta:
    """Extract method, config and optional selected_mask from a checkpoint payload."""
    if isinstance(checkpoint, dict):
        method = str(checkpoint.get("method", "lassonet"))
        config = checkpoint.get("config", {})
        selected_mask = checkpoint.get("selected_mask")
        if selected_mask is not None:
            selected_mask = np.asarray(selected_mask, dtype=bool)
        return CheckpointMeta(method=method, config=dict(config), selected_mask=selected_mask)
    return CheckpointMeta(method="lassonet", config={}, selected_mask=None)


def infer_hidden_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    """Infer LassoNet hidden_dims from the saved layer shapes."""
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


def build_model_wrapper(
    hidden_dims: Tuple[int, ...],
    meta: CheckpointMeta,
    device: torch.device,
):
    """Instantiate the correct wrapper class so the loss definition matches training."""
    common_kwargs = dict(hidden_dims=hidden_dims, device=str(device), verbose=0)
    if meta.method == "adv_before_ablation":
        return LassoNetSAMAligned(
            adv_rho=float(meta.config.get("adv_rho", 0.1)),
            adv_alpha=float(meta.config.get("adv_alpha", 1.0)),
            adv_delta=float(meta.config.get("adv_delta", 1e-12)),
            **common_kwargs,
        )
    return LassoNetClassifier(**common_kwargs)


def load_model_and_meta(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[object, nn.Module, CheckpointMeta]:
    """Load the wrapper, torch model and checkpoint metadata."""
    checkpoint = read_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    meta = extract_checkpoint_meta(checkpoint)
    hidden_dims = infer_hidden_dims(state_dict)

    wrapper = build_model_wrapper(hidden_dims, meta, device)
    wrapper.load(state_dict)
    model = wrapper.model.to(device)
    model.eval()
    return wrapper, model, meta


def align_eval_features(x_eval: np.ndarray, model: nn.Module, selected_mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply selected_mask when the checkpoint was trained on a sparse feature subset."""
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


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into one vector."""
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """Normalize a vector safely."""
    norm = vector.norm()
    if norm.item() == 0.0:
        return vector
    return vector / norm


def batch_loss(wrapper, model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """
    Reuse the project training loss definition on a batch.

    For adv_before_ablation, this mirrors the mixed clean/adversarial criterion
    used in training. The adversarial input is detached exactly as in the script.
    """
    logits = model(xb)
    crit_clean = wrapper.criterion(logits, yb)

    if isinstance(wrapper, LassoNetSAMAligned):
        adv_enabled = wrapper.adv_rho > 0 and wrapper.adv_alpha > 0
        crit = crit_clean
        if adv_enabled:
            xb_for_grad = xb.detach().requires_grad_(True)
            logits_for_grad = model(xb_for_grad)
            crit_for_grad = wrapper.criterion(logits_for_grad, yb)
            grad_x = torch.autograd.grad(crit_for_grad, xb_for_grad, retain_graph=True)[0]
            grad_norm = torch.norm(grad_x, p=2)
            perturb = wrapper.adv_rho * grad_x / (grad_norm + wrapper.adv_delta)
            xb_adv = (xb_for_grad + perturb).detach()
            crit_adv = wrapper.criterion(model(xb_adv), yb)
            crit = (1.0 - wrapper.adv_alpha) * crit_clean + wrapper.adv_alpha * crit_adv
    else:
        crit = crit_clean

    return (
        crit
        + wrapper.gamma * model.l2_regularization()
        + wrapper.gamma_skip * model.l2_regularization_skip()
    )


def hessian_vector_product(
    wrapper,
    model: nn.Module,
    params: Sequence[torch.nn.Parameter],
    loader: DataLoader,
    vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute Hessian-vector product on the subset mean loss via mini-batch accumulation."""
    vector = vector.to(device)
    total_num = 0
    hv_accum = torch.zeros_like(vector, device=device)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        batch_size = xb.size(0)
        total_num += batch_size

        loss = batch_loss(wrapper, model, xb, yb)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = flatten_tensors(grads)
        grad_dot_vec = torch.dot(flat_grads, vector)
        hv = torch.autograd.grad(grad_dot_vec, params, retain_graph=False)
        hv_flat = flatten_tensors([item.detach() for item in hv])
        hv_accum += hv_flat * (batch_size / len(loader.dataset))

    if total_num == 0:
        raise ValueError("Empty dataloader.")
    return hv_accum


def power_iteration(
    wrapper,
    model: nn.Module,
    params: Sequence[torch.nn.Parameter],
    loader: DataLoader,
    power_iters: int,
    device: torch.device,
) -> Dict[str, object]:
    """Estimate the largest Hessian eigenvalue with power iteration."""
    dim = sum(param.numel() for param in params)
    vector = normalize_vector(torch.randn(dim, device=device))
    eigenvalue_history = []

    for _ in range(power_iters):
        hv = hessian_vector_product(wrapper, model, params, loader, vector, device)
        rayleigh = float(torch.dot(vector, hv).item())
        eigenvalue_history.append(rayleigh)
        vector = normalize_vector(hv)

    hv_final = hessian_vector_product(wrapper, model, params, loader, vector, device)
    lambda_max = float(torch.dot(vector, hv_final).item())
    return {
        "lambda_max": lambda_max,
        "power_history": eigenvalue_history,
    }


def rademacher_like(vector: torch.Tensor) -> torch.Tensor:
    """Generate a Rademacher vector with entries in {-1, +1}."""
    return (torch.randint(0, 2, vector.shape, device=vector.device, dtype=torch.int64) * 2 - 1).to(
        vector.dtype
    )


def hutchinson_trace(
    wrapper,
    model: nn.Module,
    params: Sequence[torch.nn.Parameter],
    loader: DataLoader,
    hutchinson_samples: int,
    device: torch.device,
) -> Dict[str, object]:
    """Estimate Hessian trace with Hutchinson's estimator."""
    dim = sum(param.numel() for param in params)
    probe_template = torch.empty(dim, device=device, dtype=torch.float32)
    estimates = []

    for _ in range(hutchinson_samples):
        z = rademacher_like(probe_template)
        hz = hessian_vector_product(wrapper, model, params, loader, z, device)
        estimates.append(float(torch.dot(z, hz).item()))

    trace_estimate = float(np.mean(estimates)) if estimates else 0.0
    return {
        "trace": trace_estimate,
        "hutchinson_values": estimates,
    }


def lanczos_tridiagonal(
    wrapper,
    model: nn.Module,
    params: Sequence[torch.nn.Parameter],
    loader: DataLoader,
    steps: int,
    device: torch.device,
) -> Dict[str, object]:
    """Build a small Lanczos tridiagonal approximation using HVPs."""
    dim = sum(param.numel() for param in params)
    q_prev = torch.zeros(dim, device=device)
    q = normalize_vector(torch.randn(dim, device=device))
    alphas = []
    betas = []

    for step in range(max(1, steps)):
        z = hessian_vector_product(wrapper, model, params, loader, q, device)
        if step > 0:
            z = z - betas[-1] * q_prev

        alpha = torch.dot(q, z)
        alphas.append(float(alpha.item()))
        z = z - alpha * q

        beta = float(z.norm().item())
        if beta < 1e-10:
            break
        betas.append(beta)
        q_prev = q
        q = z / beta

    tri = np.diag(alphas)
    if betas:
        off_diag = np.array(betas[: len(alphas) - 1], dtype=np.float64)
        tri += np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    eigvals = np.linalg.eigvalsh(tri) if len(alphas) > 0 else np.array([], dtype=np.float64)
    return {
        "lanczos_alphas": alphas,
        "lanczos_betas": betas,
        "lanczos_eigenvalues": eigvals.tolist(),
    }


def plot_lanczos_eigenvalues(eigenvalues: List[float], path: Path) -> None:
    """Save a sorted Lanczos eigenvalue curve."""
    eigvals = np.array(sorted(eigenvalues, reverse=True), dtype=np.float64)
    plt.figure(figsize=(7, 4.5))
    if eigvals.size > 0:
        xs = np.arange(1, eigvals.size + 1)
        plt.plot(xs, eigvals, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Lanczos Hessian Eigenvalues")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_lanczos_eigenvalues_comparison(results: Sequence[Dict[str, object]], path: Path) -> None:
    """Overlay sorted Lanczos eigenvalue curves for multiple checkpoints."""
    plt.figure(figsize=(7, 4.5))
    for result in results:
        eigvals = np.array(sorted(result["lanczos_eigenvalues"], reverse=True), dtype=np.float64)
        if eigvals.size == 0:
            continue
        xs = np.arange(1, eigvals.size + 1)
        plt.plot(xs, eigvals, marker="o", linewidth=1.5, markersize=3, label=result["label"])
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Lanczos Hessian Eigenvalues Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_density(eigenvalues: List[float], path: Path) -> None:
    """Save an approximate Hessian eigenvalue density line from Lanczos eigenvalues."""
    eigvals = np.array(eigenvalues, dtype=np.float64)
    plt.figure(figsize=(7, 4.5))
    if eigvals.size > 0:
        hist_range = (eigvals.min(), eigvals.max()) if eigvals.size > 1 else (eigvals[0] - 1.0, eigvals[0] + 1.0)
        bins = min(30, max(5, eigvals.size))
        density, edges = np.histogram(eigvals, bins=bins, density=True, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, density, linewidth=2.0)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.title("Approximate Hessian Eigenvalue Density")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_density_comparison(results: Sequence[Dict[str, object]], path: Path) -> None:
    """Overlay comparable Hessian density lines using a shared histogram range."""
    eigen_sets = [np.array(result["lanczos_eigenvalues"], dtype=np.float64) for result in results]
    non_empty = [eigvals for eigvals in eigen_sets if eigvals.size > 0]

    plt.figure(figsize=(7, 4.5))
    if non_empty:
        global_min = min(float(eigvals.min()) for eigvals in non_empty)
        global_max = max(float(eigvals.max()) for eigvals in non_empty)
        if np.isclose(global_min, global_max):
            global_min -= 1.0
            global_max += 1.0
        bins = min(30, max(5, max(eigvals.size for eigvals in non_empty)))
        for result, eigvals in zip(results, eigen_sets):
            if eigvals.size == 0:
                continue
            density, edges = np.histogram(eigvals, bins=bins, density=True, range=(global_min, global_max))
            centers = 0.5 * (edges[:-1] + edges[1:])
            plt.plot(centers, density, linewidth=2.0, label=result["label"])
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.title("Approximate Hessian Eigenvalue Density Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    """Append one result row to a CSV file."""
    csv_row = {
        "timestamp": row["timestamp"],
        "label": row.get("label", ""),
        "checkpoint": row["checkpoint"],
        "dataset": row["dataset"],
        "method": row["method"],
        "subset_size": row["subset_size"],
        "batch_size": row["batch_size"],
        "power_iters": row["power_iters"],
        "hutchinson_samples": row["hutchinson_samples"],
        "feature_count": row["feature_count"],
        "lambda_max": row["lambda_max"],
        "trace": row["trace"],
        "average_curvature": row["average_curvature"],
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
    dataset: str,
    subset_size: int,
    batch_size: int,
    power_iters: int,
    hutchinson_samples: int,
    lanczos_steps: int,
    device: torch.device,
) -> Dict[str, object]:
    """Run Hessian analysis for one checkpoint and package the result."""
    x_eval, y_eval = load_eval_subset(dataset, subset_size)
    wrapper, model, meta = load_model_and_meta(checkpoint_path, device)
    x_eval = align_eval_features(x_eval, model, meta.selected_mask)
    loader = build_eval_loader(x_eval, y_eval, batch_size)
    params = trainable_parameters(model)
    feature_count = int(x_eval.shape[1])

    power_stats = power_iteration(
        wrapper,
        model,
        params,
        loader,
        power_iters=power_iters,
        device=device,
    )
    trace_stats = hutchinson_trace(
        wrapper,
        model,
        params,
        loader,
        hutchinson_samples=hutchinson_samples,
        device=device,
    )
    lanczos_stats = lanczos_tridiagonal(
        wrapper,
        model,
        params,
        loader,
        steps=lanczos_steps,
        device=device,
    )

    trace_value = float(trace_stats["trace"])
    avg_curvature = float(trace_value / feature_count) if feature_count > 0 else 0.0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "timestamp": timestamp,
        "label": label,
        "checkpoint": str(checkpoint_path),
        "dataset": canonical_dataset_name(dataset),
        "method": meta.method,
        "subset_size": int(len(loader.dataset)),
        "batch_size": int(batch_size),
        "power_iters": int(power_iters),
        "hutchinson_samples": int(hutchinson_samples),
        "lanczos_steps": int(lanczos_steps),
        "device": str(device),
        "feature_count": feature_count,
        "lambda_max": float(power_stats["lambda_max"]),
        "trace": trace_value,
        "average_curvature": avg_curvature,
        "power_history": power_stats["power_history"],
        "hutchinson_values": trace_stats["hutchinson_values"],
        "lanczos_eigenvalues": lanczos_stats["lanczos_eigenvalues"],
        "lanczos_alphas": lanczos_stats["lanczos_alphas"],
        "lanczos_betas": lanczos_stats["lanczos_betas"],
    }


def main() -> None:
    """Run Hessian spectrum analysis and save results."""
    args = parse_hessian_args()
    device = resolve_device(args.device)
    checkpoint_paths = [Path(path) for path in args.checkpoint]
    labels = normalize_labels(checkpoint_paths, args.label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "hessian_spectrum_summary.csv"
    compare_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: List[Dict[str, object]] = []

    for checkpoint_path, label in zip(checkpoint_paths, labels):
        result = analyze_checkpoint(
            checkpoint_path,
            label,
            dataset=args.dataset,
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            power_iters=args.power_iters,
            hutchinson_samples=args.hutchinson_samples,
            lanczos_steps=args.lanczos_steps,
            device=device,
        )
        stem = safe_stem(checkpoint_path)
        prefix = f"{stem}_hessian_{result['timestamp']}"
        json_path = output_dir / f"{prefix}.json"
        eig_curve_path = output_dir / f"{prefix}_lanczos_curve.png"
        density_path = output_dir / f"{prefix}_density.png"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        append_csv_row(csv_path, result)
        plot_lanczos_eigenvalues(result["lanczos_eigenvalues"], eig_curve_path)
        plot_density(result["lanczos_eigenvalues"], density_path)

        results.append(result)
        print(f"Saved JSON: {json_path}")
        print(f"Saved Lanczos eigenvalue curve PNG: {eig_curve_path}")
        print(f"Saved Hessian density PNG: {density_path}")
        print(
            f"[{label}] Metrics: "
            f"lambda_max={result['lambda_max']:.6f}, "
            f"trace={result['trace']:.6f}, "
            f"average_curvature={result['average_curvature']:.6e}"
        )

    print(f"Appended CSV: {csv_path}")

    if len(results) > 1:
        joined_labels = "_vs_".join(label.replace(" ", "_") for label in labels)
        compare_curve_path = output_dir / f"{joined_labels}_hessian_compare_{compare_timestamp}_lanczos_curve.png"
        compare_density_path = output_dir / f"{joined_labels}_hessian_compare_{compare_timestamp}_density.png"
        compare_json = output_dir / f"{joined_labels}_hessian_compare_{compare_timestamp}.json"

        plot_lanczos_eigenvalues_comparison(results, compare_curve_path)
        plot_density_comparison(results, compare_density_path)

        with compare_json.open("w", encoding="utf-8") as f:
            json.dump({"timestamp": compare_timestamp, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"Saved comparison JSON: {compare_json}")
        print(f"Saved comparison Lanczos curve PNG: {compare_curve_path}")
        print(f"Saved comparison density PNG: {compare_density_path}")


if __name__ == "__main__":
    main()
