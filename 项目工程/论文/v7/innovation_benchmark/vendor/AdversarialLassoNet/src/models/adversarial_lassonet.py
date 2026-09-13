import argparse
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lassonet import LassoNetClassifier
from lassonet.interfaces import HistoryItem
from lassonet.utils import eval_on_path


DATASET_ALIASES = {
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

ALL_DATASETS = [
    "MNIST",
    "ColoredMNIST",
    "MNIST-Fashion",
    "MICE",
    "COIL",
    "ISOLET",
    "Activity",
]


class AdversarialLassoNetClassifier(LassoNetClassifier):
    """Enhanced adversarial-perturbation LassoNet with stability loss, gradient-norm penalty, and prox adjustment."""

    def __init__(
        self,
        adv_rho=0.1,
        adv_alpha=0.0,
        adv_delta=1e-12,
        stability_weight=0.0,
        grad_norm_weight=0.0,
        prox_lambda_bar_ratio=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adv_rho = adv_rho
        self.adv_alpha = adv_alpha
        self.adv_delta = adv_delta
        self.stability_weight = stability_weight
        self.grad_norm_weight = grad_norm_weight
        self.prox_lambda_bar_ratio = prox_lambda_bar_ratio

    @staticmethod
    def _grad_norm_penalty(grad: torch.Tensor) -> torch.Tensor:
        grad_flat = grad.reshape(grad.shape[0], -1)
        return grad_flat.norm(p=2, dim=1).mean()

    def _train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        batch_size,
        epochs,
        lambda_,
        optimizer,
        return_state_dict,
        patience=None,
    ) -> HistoryItem:
        model = self.model

        def validation_obj():
            with torch.no_grad():
                return (
                    self.criterion(model(X_val), y_val).item()
                    + lambda_ * model.l1_regularization_skip().item()
                    + self.gamma * model.l2_regularization().item()
                    + self.gamma_skip * model.l2_regularization_skip().item()
                )

        best_val_obj = validation_obj()
        epochs_since_best_val_obj = 0
        if self.backtrack:
            best_state_dict = self.model.cpu_state_dict()
            real_best_val_obj = best_val_obj
            real_loss = float("nan")

        n_iters = 0
        n_train = len(X_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)

        pert_enabled = self.adv_rho > 0 and (
            self.adv_alpha > 0 or self.stability_weight > 0
        )
        grad_penalty_enabled = self.grad_norm_weight > 0
        need_input_grad = pert_enabled or grad_penalty_enabled

        for epoch in tqdm(
            range(epochs),
            desc=f"Training LassoNet Path (lambda={lambda_:.4f})",
            leave=False,
            dynamic_ncols=True,
            disable=self.verbose == 0,
        ):
            indices = randperm(n_train)
            model.train()
            loss = 0.0
            for start in range(0, n_train, batch_size):
                batch = indices[start : start + batch_size]

                def closure():
                    nonlocal loss
                    optimizer.zero_grad()

                    xb = X_train[batch]
                    yb = y_train[batch]

                    if need_input_grad:
                        xb = xb.detach().requires_grad_(True)

                    logits_clean = model(xb)
                    crit_clean = self.criterion(logits_clean, yb)
                    crit = crit_clean
                    grad = None

                    if need_input_grad:
                        grad = torch.autograd.grad(
                            crit_clean,
                            xb,
                            retain_graph=True,
                            create_graph=grad_penalty_enabled,
                        )[0]

                    if pert_enabled and grad is not None:
                        grad_for_perturb = grad.detach()
                        grad_norm = torch.norm(grad_for_perturb, p=2)
                        perturb = self.adv_rho * grad_for_perturb / (
                            grad_norm + self.adv_delta
                        )
                        xb_adv = (xb + perturb).detach()
                        logits_adv = model(xb_adv)

                        if self.adv_alpha > 0:
                            crit_adv = self.criterion(logits_adv, yb)
                            crit = crit + self.adv_alpha * crit_adv

                        if self.stability_weight > 0:
                            stability = F.mse_loss(logits_adv, logits_clean.detach())
                            crit = crit + self.stability_weight * stability

                    if grad_penalty_enabled and grad is not None:
                        crit = crit + self.grad_norm_weight * self._grad_norm_penalty(grad)

                    ans = (
                        crit
                        + self.gamma * model.l2_regularization()
                        + self.gamma_skip * model.l2_regularization_skip()
                    )
                    if not torch.isfinite(ans):
                        print(f"Loss is {ans}", file=sys.stderr)
                        print("Did you normalize input?", file=sys.stderr)
                        print("Loss::", crit.item())
                        print("l2_regularization:", model.l2_regularization())
                        print("l2_regularization_skip:", model.l2_regularization_skip())
                        assert False
                    ans.backward()
                    loss += ans.item() * len(batch) / n_train
                    return ans

                optimizer.step(closure)
                prox_step = lambda_ * optimizer.param_groups[0]["lr"]
                model.prox(
                    lambda_=prox_step,
                    lambda_bar=prox_step * self.prox_lambda_bar_ratio,
                    M=self.M,
                )

            if epoch == 0:
                real_loss = loss
            model.eval()
            val_obj = validation_obj()
            if val_obj < self.tol * best_val_obj:
                best_val_obj = val_obj
                epochs_since_best_val_obj = 0
            else:
                epochs_since_best_val_obj += 1

            if self.backtrack and val_obj < real_best_val_obj:
                best_state_dict = self.model.cpu_state_dict()
                real_best_val_obj = val_obj
                real_loss = loss
                n_iters = epoch + 1

            if patience is not None and epochs_since_best_val_obj == patience:
                break

        if self.backtrack:
            self.model.load_state_dict(best_state_dict)
            val_obj = real_best_val_obj
            loss = real_loss
        else:
            n_iters = epoch + 1

        with torch.no_grad():
            reg = self.model.l1_regularization_skip().item()
            l2_regularization = self.model.l2_regularization()
            l2_regularization_skip = self.model.l2_regularization_skip()

        return HistoryItem(
            lambda_=lambda_,
            state_dict=self.model.cpu_state_dict() if return_state_dict else None,
            objective=loss + lambda_ * reg,
            loss=loss,
            val_objective=val_obj,
            val_loss=val_obj - lambda_ * reg,
            regularization=reg,
            l2_regularization=l2_regularization,
            l2_regularization_skip=l2_regularization_skip,
            selected=self.model.input_mask().cpu(),
            n_iters=n_iters,
        )


# Backward-compatible alias used by older experiment scripts.
LassoNetSAMAligned = AdversarialLassoNetClassifier
ImprovedAdvLassoNetClassifier = AdversarialLassoNetClassifier
RobustFeatureSelectionLassoNetClassifier = AdversarialLassoNetClassifier


@dataclass(frozen=True)
class AdversarialLassoNetConfig:
    k: int = 50
    lasso_epochs: int = 1000
    batch_size: int = 512
    lasso_verbose: int = 0
    M: int = 10
    adv_rho: float = 0.1
    adv_alpha: float = 1.0
    adv_delta: float = 1e-12
    stability_weight: float = 1.0
    grad_norm_weight: float = 0.1
    prox_lambda_bar_ratio: float = 1.0


@dataclass
class FeatureSelectionRunResult:
    selected_count: int
    val_acc: float
    test_acc: float
    selected_mask: np.ndarray
    path_sparse: List[HistoryItem]
    selected_lambda: float


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.lower(), name)


def resolve_datasets(text: str) -> List[str]:
    if text.lower() == "all":
        return ALL_DATASETS
    return [canonical_dataset_name(token.strip()) for token in text.split(",") if token.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adversarial_lassonet(
    input_dim: int,
    seed: int,
    config: AdversarialLassoNetConfig,
    device: torch.device,
) -> AdversarialLassoNetClassifier:
    hidden_dim = (max(1, input_dim // 3),)
    return AdversarialLassoNetClassifier(
        M=config.M,
        hidden_dims=hidden_dim,
        verbose=config.lasso_verbose,
        torch_seed=seed,
        random_state=seed,
        device=str(device),
        n_iters=config.lasso_epochs,
        batch_size=config.batch_size,
        adv_rho=config.adv_rho,
        adv_alpha=config.adv_alpha,
        adv_delta=config.adv_delta,
        stability_weight=config.stability_weight,
        grad_norm_weight=config.grad_norm_weight,
        prox_lambda_bar_ratio=config.prox_lambda_bar_ratio,
    )


def train_adversarial_lassonet_once(
    *,
    dataset: str,
    seed: int,
    device: torch.device,
    config: AdversarialLassoNetConfig,
    load_dataset_fn,
) -> FeatureSelectionRunResult:
    set_seed(seed)

    loaded = load_dataset_fn(dataset)
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

    selector = build_adversarial_lassonet(
        X_train.shape[1], seed, config, device
    )
    path = selector.path(X_train, y_train, X_val=X_val, y_val=y_val)

    desired = min(path, key=lambda save: abs(int(save.selected.sum().item()) - config.k))
    selected_mask = desired.selected.cpu().numpy().astype(bool)
    selected_count = int(selected_mask.sum())
    if selected_count == 0:
        raise RuntimeError(
            f"{dataset} selected 0 features. Please lower --k or reduce the sparsity strength."
        )

    X_train_selected = X_train[:, selected_mask]
    X_val_selected = X_val[:, selected_mask]
    X_test_selected = X_test[:, selected_mask]

    refit = build_adversarial_lassonet(
        X_train_selected.shape[1], seed, config, device
    )
    path_sparse = refit.path(
        X_train_selected,
        y_train,
        X_val=X_val_selected,
        y_val=y_val,
        return_state_dicts=True,
    )[:1]

    val_acc = float(eval_on_path(refit, path_sparse, X_val_selected, y_val)[0])
    test_acc = float(eval_on_path(refit, path_sparse, X_test_selected, y_test)[0])
    return FeatureSelectionRunResult(
        selected_count=selected_count,
        val_acc=val_acc,
        test_acc=test_acc,
        selected_mask=selected_mask,
        path_sparse=path_sparse,
        selected_lambda=float(desired.lambda_),
    )


def save_sparse_checkpoint(
    *,
    output_dir: str,
    dataset: str,
    seed: int,
    result: FeatureSelectionRunResult,
    config: AdversarialLassoNetConfig,
    method: str = "improved_adv_lassonet",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{method}_path_seed{seed}.pkl"
    payload = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "selected_mask": result.selected_mask.astype(bool),
        "selected_count": result.selected_count,
        "selected_lambda": result.selected_lambda,
        "path_sparse": result.path_sparse,
        "config": {
            "k": config.k,
            "lasso_epochs": config.lasso_epochs,
            "batch_size": config.batch_size,
            "adv_rho": config.adv_rho,
            "adv_alpha": config.adv_alpha,
            "adv_delta": config.adv_delta,
            "stability_weight": config.stability_weight,
            "grad_norm_weight": config.grad_norm_weight,
            "prox_lambda_bar_ratio": config.prox_lambda_bar_ratio,
        },
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return out_path


def summary_to_jsonable(summary: Dict[str, object]) -> Dict[str, object]:
    ans = dict(summary)
    selected_mask = ans.get("selected_mask")
    if isinstance(selected_mask, np.ndarray):
        ans["selected_mask"] = selected_mask.astype(bool).tolist()
    ans.pop("path_sparse", None)
    return ans


def namespace_to_adversarial_lassonet_config(
    args: argparse.Namespace,
) -> AdversarialLassoNetConfig:
    return AdversarialLassoNetConfig(
        k=args.k,
        lasso_epochs=args.lasso_epochs,
        batch_size=args.batch_size,
        lasso_verbose=args.lasso_verbose,
        M=getattr(args, "M", 10),
        adv_rho=args.adv_rho,
        adv_alpha=args.adv_alpha,
        adv_delta=args.adv_delta,
        stability_weight=getattr(args, "stability_weight", 1.0),
        grad_norm_weight=getattr(args, "grad_norm_weight", 0.1),
        prox_lambda_bar_ratio=getattr(args, "prox_lambda_bar_ratio", 1.0),
    )


def build_adv_arg_parser(
    description: Optional[str] = None,
    *,
    include_dataset_list: bool = True,
    include_runs: bool = True,
    include_save_json: bool = True,
    include_save_pkl_dir: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or "Improved adversarial-perturbation LassoNet experiment runner."
    )
    parser.add_argument("--dataset", type=str, default="", help='Train one dataset, e.g. "mice".')
    if include_dataset_list:
        parser.add_argument(
            "--datasets",
            type=str,
            default="all",
            help='Comma-separated list or "all".',
        )
    if include_runs:
        parser.add_argument("--runs", type=int, default=5)
        parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50, help="Target selected feature count.")
    parser.add_argument("--lasso-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lasso-verbose", type=int, default=0)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--adv-rho", type=float, default=0.1)
    parser.add_argument("--adv-alpha", type=float, default=1.0)
    parser.add_argument("--adv-delta", type=float, default=1e-12)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--grad-norm-weight", type=float, default=0.1)
    parser.add_argument("--prox-lambda-bar-ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    if include_save_json:
        parser.add_argument("--save-json", type=str, default="", help="Optional output json path.")
    if include_save_pkl_dir:
        parser.add_argument(
            "--save-pkl-dir",
            type=str,
            default="",
            help="Optional directory for sparse checkpoint pkl files.",
        )
    return parser


# Backward-compatible exported names for existing scripts.
AdvLassoNetConfig = AdversarialLassoNetConfig
RunResult = FeatureSelectionRunResult
build_improved_adv_lassonet = build_adversarial_lassonet
train_improved_adv_lassonet_once = train_adversarial_lassonet_once
namespace_to_adv_config = namespace_to_adversarial_lassonet_config
RobustLassoNetExperimentConfig = AdversarialLassoNetConfig
build_robust_feature_selection_lassonet = build_adversarial_lassonet
train_robust_feature_selection_lassonet_once = train_adversarial_lassonet_once
namespace_to_robust_lassonet_config = namespace_to_adversarial_lassonet_config
