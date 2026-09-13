import copy
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    val_loss: float
    selected: torch.BoolTensor
    n_iters: int


class _GatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(input_dim))

        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x * self.gate)

    def selected_mask(self) -> torch.BoolTensor:
        return self.gate.detach().abs() > 1e-8


class FISTATabularClassifier(BaseEstimator, ClassifierMixin):
    """
    FISTA-inspired feature selection baseline for tabular classification.

    It is not the original imaging FISTA-Net architecture from the paper.
    This version adapts the proximal-gradient + momentum idea to a gated MLP
    so it can be benchmarked on tabular datasets with the same protocol used
    by LassoNet in this repository.
    """

    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...] = (100,),
        lambda_start: float = 1e-4,
        path_multiplier: float = 1.5,
        lambda_seq: Optional[Iterable[float]] = None,
        n_iters_init: int = 200,
        n_iters_path: int = 100,
        patience_init: int = 30,
        patience_path: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_size: float = 0.1,
        device: Optional[str] = None,
        verbose: int = 1,
        random_state: Optional[int] = None,
        torch_seed: Optional[int] = None,
    ):
        self.hidden_dims = hidden_dims
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.lambda_seq = lambda_seq
        self.n_iters_init = n_iters_init
        self.n_iters_path = n_iters_path
        self.patience_init = patience_init
        self.patience_path = patience_path
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_size = val_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.random_state = random_state
        self.torch_seed = torch_seed

        self.model: Optional[_GatedMLP] = None
        self.path_: List[HistoryItem] = []

    def _set_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.torch_seed)

    def _cast_input(self, X, y=None):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if y is None:
            return X
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return X, y

    def _init_model(self, X, y):
        self._set_seed()
        output_dim = int(np.max(y)) + 1
        self.model = _GatedMLP(X.shape[1], self.hidden_dims, output_dim).to(self.device)

    def _batch_indices(self, n_samples: int):
        if self.batch_size is None or self.batch_size >= n_samples:
            yield torch.arange(n_samples, device=self.device)
            return

        indices = torch.randperm(n_samples, device=self.device)
        for start in range(0, n_samples, self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) > 0:
                yield batch

    def _evaluate_loss(self, X, y) -> float:
        self.model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(self.model(X), y)
        return float(loss.item())

    def _train_stage(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        lambda_,
        epochs,
        patience,
        return_state_dict,
    ) -> HistoryItem:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_state = None
        epochs_since_best = 0
        performed_epochs = 0

        gate_prev = self.model.gate.detach().clone()
        t_prev = 1.0

        for epoch in range(epochs):
            self.model.train()
            for batch in self._batch_indices(len(X_train)):
                optimizer.zero_grad()
                logits = self.model(X_train[batch])
                loss = F.cross_entropy(logits, y_train[batch])
                loss.backward()
                optimizer.step()

                # FISTA-style proximal step on the feature gate.
                with torch.no_grad():
                    gate_after_grad = self.model.gate.detach().clone()
                    prox_gate = torch.sign(gate_after_grad) * torch.relu(
                        gate_after_grad.abs() - self.lr * lambda_
                    )
                    t_new = (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0
                    momentum = (t_prev - 1.0) / t_new
                    accelerated_gate = prox_gate + momentum * (prox_gate - gate_prev)
                    self.model.gate.copy_(accelerated_gate)
                    gate_prev = prox_gate
                    t_prev = t_new

            val_loss = self._evaluate_loss(X_val, y_val)
            performed_epochs = epoch + 1
            if val_loss < best_val:
                best_val = val_loss
                epochs_since_best = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_since_best += 1
                if patience is not None and epochs_since_best >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return HistoryItem(
            lambda_=lambda_,
            state_dict=copy.deepcopy(self.model.state_dict()) if return_state_dict else None,
            val_loss=best_val,
            selected=self.model.selected_mask().detach().cpu(),
            n_iters=performed_epochs,
        )

    def path(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        lambda_seq=None,
        return_state_dicts=False,
    ) -> List[HistoryItem]:
        assert (X_val is None) == (
            y_val is None
        ), "X_val and y_val must either both be provided or both be omitted"

        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y

        X_train, y_train = self._cast_input(X_train, y_train)
        X_val, y_val = self._cast_input(X_val, y_val)

        self._init_model(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
        self.path_ = []

        init_item = self._train_stage(
            X_train,
            y_train,
            X_val,
            y_val,
            lambda_=0.0,
            epochs=self.n_iters_init,
            patience=self.patience_init,
            return_state_dict=return_state_dicts,
        )
        self.path_.append(init_item)

        if lambda_seq is None:
            if self.lambda_seq is not None:
                lambda_seq = self.lambda_seq
            else:
                current = self.lambda_start
                seq = []
                while len(seq) < 40:
                    seq.append(current)
                    current *= self.path_multiplier
                lambda_seq = seq

        for current_lambda in lambda_seq:
            item = self._train_stage(
                X_train,
                y_train,
                X_val,
                y_val,
                lambda_=float(current_lambda),
                epochs=self.n_iters_path,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            self.path_.append(item)
            if self.verbose > 1:
                print(
                    f"lambda={current_lambda:.3e}, selected={int(item.selected.sum().item())}, val_loss={item.val_loss:.6f}"
                )
            if not item.selected.any():
                break

        return self.path_

    def load(self, state_dict):
        if self.model is None:
            gate = state_dict["gate"]
            output_dim = next(
                tensor.shape[0]
                for name, tensor in reversed(list(state_dict.items()))
                if name.endswith(".bias")
            )
            self.model = _GatedMLP(len(gate), self.hidden_dims, output_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        return self

    def fit(self, X, y, *, X_val=None, y_val=None):
        self.path(X, y, X_val=X_val, y_val=y_val, return_state_dicts=False)
        return self

    def predict(self, X):
        self.model.eval()
        X = self._cast_input(X)
        with torch.no_grad():
            pred = self.model(X).argmax(dim=1)
        return pred.cpu().numpy()

    def score(self, X, y):
        pred = self.predict(X)
        return float((pred == np.asarray(y)).mean())
