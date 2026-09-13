import copy
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.autograd as autograd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_dimension_glasso(var: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return var.pow(2).sum(dim=dim).add(1e-8).sqrt().sum()


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = (input_dim,) + hidden_dims + (output_dim,)
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainResult:
    state_dict: dict[str, torch.Tensor]
    train_history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    best_val_acc: float


@dataclass
class DeepLassoRunResult:
    seed: int
    dataset: str
    task: str
    selected_count: int
    target_k: int
    selector_best_epoch: int
    selector_best_val_loss: float
    selector_best_val_acc: float
    refit_best_epoch: int
    refit_val_loss: float
    refit_val_score: float
    test_loss: float
    test_score: float
    selected_indices: list[int]
    selected_mask: np.ndarray
    importance: np.ndarray
    selector_history: list[dict[str, float]]
    refit_history: list[dict[str, float]]
    selector_state_dict: dict[str, torch.Tensor]
    refit_state_dict: dict[str, torch.Tensor]


def infer_hidden_dims(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    selected_count: Optional[int] = None,
) -> tuple[int, ...]:
    width_base = selected_count if selected_count is not None else input_dim
    width = hidden_dim if hidden_dim > 0 else max(8, width_base // 3)
    return tuple(width for _ in range(max(1, depth)))


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    criterion: nn.Module,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_count += batch_y.size(0)

    return total_loss / total_count, total_correct / total_count


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    reg_weight: float,
    regularization: Optional[str],
    selection_metric: str,
) -> TrainResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    history = []

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = -float("inf")
    best_metric = float("inf") if selection_metric == "val_loss" else -float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_task = 0.0
        epoch_reg = 0.0
        total_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            if regularization == "deep_lasso":
                batch_x.requires_grad_(True)

            logits = model(batch_x)
            task_loss = criterion(logits, batch_y)

            if regularization == "deep_lasso":
                grad_inputs = autograd.grad(
                    task_loss,
                    batch_x,
                    create_graph=True,
                    allow_unused=False,
                )[0]
                reg = add_dimension_glasso(grad_inputs, dim=0)
                loss = reg_weight * reg + (1.0 - reg_weight) * task_loss
            else:
                reg = torch.zeros((), device=device)
                loss = task_loss

            loss.backward()
            optimizer.step()

            batch_size_now = batch_y.size(0)
            epoch_total += loss.item() * batch_size_now
            epoch_task += task_loss.item() * batch_size_now
            epoch_reg += reg.item() * batch_size_now
            total_count += batch_size_now

        val_loss, val_acc = evaluate(model, X_val, y_val, device, criterion, batch_size)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": epoch_total / total_count,
                "train_task_loss": epoch_task / total_count,
                "train_reg": epoch_reg / total_count,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        current_metric = val_loss if selection_metric == "val_loss" else val_acc
        improved = current_metric < best_metric if selection_metric == "val_loss" else current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

    return TrainResult(
        state_dict=best_state_dict,
        train_history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )


def compute_deep_lasso_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    grads = []
    model.eval()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x.requires_grad_(True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        grad_inputs = autograd.grad(
            loss,
            batch_x,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads.append(grad_inputs.detach().cpu())

    grad_matrix = torch.cat(grads, dim=0)
    importance = grad_matrix.pow(2).sum(dim=0).sqrt()
    return importance.numpy()


def select_top_k(importance: np.ndarray, k: int) -> np.ndarray:
    feature_count = importance.shape[0]
    k = max(1, min(int(k), feature_count))
    indices = np.argpartition(-importance, kth=k - 1)[:k]
    mask = np.zeros(feature_count, dtype=bool)
    mask[indices] = True
    return mask


def load_state_dict_to_model(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict(state_dict)
    model.to(device)


class DeepLassoTabularPipeline:
    def __init__(
        self,
        *,
        hidden_dim: int = 0,
        depth: int = 2,
        dropout: float = 0.0,
        epochs: int = 200,
        refit_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        reg_weight: float = 0.2,
        selection_metric: str = "val_loss",
        device: Optional[torch.device] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.refit_epochs = refit_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.reg_weight = reg_weight
        self.selection_metric = selection_metric
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        dataset: str,
        seed: int,
        X_train_valid: np.ndarray,
        y_train_valid: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int,
    ) -> DeepLassoRunResult:
        set_seed(seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_valid,
            y_train_valid,
            test_size=0.125,
            random_state=seed,
            stratify=y_train_valid if len(np.unique(y_train_valid)) > 1 else None,
        )

        num_classes = int(max(y_train_valid.max(), y_test.max()) + 1)
        selector_hidden = infer_hidden_dims(X_train.shape[1], self.hidden_dim, self.depth)
        selector = TabularMLP(
            input_dim=X_train.shape[1],
            output_dim=num_classes,
            hidden_dims=selector_hidden,
            dropout=self.dropout,
        ).to(self.device)

        selector_result = train_model(
            model=selector,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            reg_weight=self.reg_weight,
            regularization="deep_lasso",
            selection_metric=self.selection_metric,
        )
        load_state_dict_to_model(selector, selector_result.state_dict, self.device)

        importance = compute_deep_lasso_importance(
            model=selector,
            X=X_val,
            y=y_val,
            device=self.device,
            batch_size=self.batch_size,
        )
        selected_mask = select_top_k(importance, k)
        selected_count = int(selected_mask.sum())
        selected_indices = np.flatnonzero(selected_mask).tolist()

        X_train_selected = X_train[:, selected_mask]
        X_val_selected = X_val[:, selected_mask]
        X_test_selected = X_test[:, selected_mask]

        refit_hidden = infer_hidden_dims(
            X_train.shape[1],
            self.hidden_dim,
            self.depth,
            selected_count=selected_count,
        )
        refit = TabularMLP(
            input_dim=selected_count,
            output_dim=num_classes,
            hidden_dims=refit_hidden,
            dropout=self.dropout,
        ).to(self.device)
        refit_result = train_model(
            model=refit,
            X_train=X_train_selected,
            y_train=y_train,
            X_val=X_val_selected,
            y_val=y_val,
            device=self.device,
            epochs=self.refit_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            reg_weight=0.0,
            regularization=None,
            selection_metric="val_loss",
        )
        load_state_dict_to_model(refit, refit_result.state_dict, self.device)

        criterion = nn.CrossEntropyLoss()
        refit_val_loss, refit_val_acc = evaluate(
            refit,
            X_val_selected,
            y_val,
            self.device,
            criterion,
            self.batch_size,
        )
        test_loss, test_acc = evaluate(
            refit,
            X_test_selected,
            y_test,
            self.device,
            criterion,
            self.batch_size,
        )

        return DeepLassoRunResult(
            seed=seed,
            dataset=dataset,
            task="multiclass",
            selected_count=selected_count,
            target_k=int(k),
            selector_best_epoch=selector_result.best_epoch,
            selector_best_val_loss=selector_result.best_val_loss,
            selector_best_val_acc=selector_result.best_val_acc,
            refit_best_epoch=refit_result.best_epoch,
            refit_val_loss=refit_val_loss,
            refit_val_score=refit_val_acc,
            test_loss=test_loss,
            test_score=test_acc,
            selected_indices=selected_indices,
            selected_mask=selected_mask,
            importance=importance,
            selector_history=selector_result.train_history,
            refit_history=refit_result.train_history,
            selector_state_dict=selector_result.state_dict,
            refit_state_dict=refit_result.state_dict,
        )
