import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 256
    lr: float = 1e-3
    num_epochs: int = 10
    hidden_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Standard Colored MNIST setup:
    # first inject label noise, then correlate color with the noisy label.
    label_flip_prob: float = 0.25

    # More training environments make it easier to test whether the model
    # is actually learning stable features.
    train_env_probs: Tuple[float, ...] = (0.1, 0.2, 0.3)
    id_env_prob: float = 0.2
    ood_env_prob: float = 0.9
    intervention_env_prob: float = 0.5

    irm_penalty_weight: float = 1000.0
    irm_penalty_anneal_iters: int = 100

    train_subset_per_env: int = 12000
    eval_subset: int = 10000
    num_workers: int = 2


class ColoredMNISTEnv(Dataset):
    """
    Build one Colored MNIST environment.

    In causal terms, this can be interpreted as:
    digit -> label
    label -> observed noisy_label
    noisy_label -> color

    Here, color only exhibits different correlations with the label across
    environments, so it is a typical spurious feature, while digit shape is
    the relatively stable feature.
    """

    def __init__(
        self,
        mnist_images: torch.Tensor,
        mnist_labels: torch.Tensor,
        env_flip_prob: float,
        indices: np.ndarray,
        label_flip_prob: float,
    ):
        super().__init__()

        images = mnist_images[indices].float() / 255.0
        digits = mnist_labels[indices]

        y_true = (digits >= 5).long()
        label_flip = torch.bernoulli(
            torch.full_like(y_true.float(), label_flip_prob)
        ).long()
        y = y_true ^ label_flip

        color_flip = torch.bernoulli(
            torch.full_like(y.float(), env_flip_prob)
        ).long()
        color = y ^ color_flip

        x = torch.zeros(images.size(0), 2, 28, 28)
        x[torch.arange(images.size(0)), color, :, :] = images

        self.x = x
        self.y = y
        self.y_true = y_true
        self.color = color
        self.env_flip_prob = env_flip_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.y_true[idx], self.color[idx]


def load_mnist_train_test(data_root: str):
    train_mnist = datasets.MNIST(root=data_root, train=True, download=True)
    test_mnist = datasets.MNIST(root=data_root, train=False, download=True)
    return (
        train_mnist.data.clone(),
        train_mnist.targets.clone(),
        test_mnist.data.clone(),
        test_mnist.targets.clone(),
    )


def make_env_datasets(cfg: Config):
    train_images, train_labels, test_images, test_labels = load_mnist_train_test(
        cfg.data_root
    )

    perm = np.random.permutation(len(train_labels))
    train_envs = []
    start = 0
    for p in cfg.train_env_probs:
        end = start + cfg.train_subset_per_env
        idx = perm[start:end]
        start = end
        train_envs.append(
            ColoredMNISTEnv(
                train_images,
                train_labels,
                env_flip_prob=p,
                indices=idx,
                label_flip_prob=cfg.label_flip_prob,
            )
        )

    eval_size = min(cfg.eval_subset, len(test_labels))
    eval_perm = np.random.permutation(len(test_labels))
    shared_eval_idx = eval_perm[:eval_size]

    id_env = ColoredMNISTEnv(
        test_images,
        test_labels,
        env_flip_prob=cfg.id_env_prob,
        indices=shared_eval_idx,
        label_flip_prob=cfg.label_flip_prob,
    )
    ood_env = ColoredMNISTEnv(
        test_images,
        test_labels,
        env_flip_prob=cfg.ood_env_prob,
        indices=shared_eval_idx,
        label_flip_prob=cfg.label_flip_prob,
    )
    intervention_env = ColoredMNISTEnv(
        test_images,
        test_labels,
        env_flip_prob=cfg.intervention_env_prob,
        indices=shared_eval_idx,
        label_flip_prob=cfg.label_flip_prob,
    )

    return train_envs, {"id": id_env, "ood": ood_env, "intervention": intervention_env}


class SmallCNN(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.featurizer = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.featurizer(x)
        return self.classifier(feat).squeeze(1)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_num = 0
    total_loss = 0.0
    total_correct_noisy = 0
    total_correct_true = 0

    if len(loader.dataset) == 0:
        raise ValueError("Evaluation dataset is empty. Check eval_subset and dataset splitting.")

    with torch.no_grad():
        for x, y, y_true, _ in loader:
            x = x.to(device)
            y = y.float().to(device)
            y_true = y_true.long().to(device)

            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).long()

            batch_size = y.size(0)
            total_num += batch_size
            total_loss += loss.item() * batch_size
            total_correct_noisy += (preds == y.long()).sum().item()
            total_correct_true += (preds == y_true).sum().item()

    return {
        "loss": total_loss / total_num,
        "acc_noisy": total_correct_noisy / total_num,
        "acc_true": total_correct_true / total_num,
    }


def evaluate_color_sensitivity(
    model: nn.Module, loader: DataLoader, device: str
) -> float:
    model.eval()
    changed = 0
    total = 0

    if len(loader.dataset) == 0:
        raise ValueError("Evaluation dataset is empty. Check eval_subset and dataset splitting.")

    with torch.no_grad():
        for x, _, _, _ in loader:
            x = x.to(device)

            pred1 = (torch.sigmoid(model(x)) > 0.5).long()
            x_flip = x.clone()
            x_flip[:, 0, :, :], x_flip[:, 1, :, :] = x[:, 1, :, :], x[:, 0, :, :]
            pred2 = (torch.sigmoid(model(x_flip)) > 0.5).long()

            changed += (pred1 != pred2).sum().item()
            total += x.size(0)

    return changed / total


def evaluate_all(
    model: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    device: str,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, loader in eval_loaders.items():
        metrics = evaluate(model, loader, device)
        metrics["color_sensitivity"] = evaluate_color_sensitivity(model, loader, device)
        results[name] = metrics
    return results


def format_eval_results(prefix: str, results: Dict[str, Dict[str, float]]) -> str:
    parts = [prefix]
    for env_name in ("id", "ood", "intervention"):
        m = results[env_name]
        parts.append(
            f"{env_name.upper()} true={m['acc_true']:.4f} "
            f"noisy={m['acc_noisy']:.4f} "
            f"flip={m['color_sensitivity']:.4f}"
        )
    return " | ".join(parts)


def train_erm(
    model: nn.Module,
    train_loaders,
    eval_loaders: Dict[str, DataLoader],
    cfg: Config,
):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("\n===== Training ERM =====")
    for epoch in range(cfg.num_epochs):
        model.train()
        env_iters = [iter(loader) for loader in train_loaders]
        steps_per_epoch = min(len(loader) for loader in train_loaders)

        for _ in range(steps_per_epoch):
            all_x = []
            all_y = []

            for it in env_iters:
                x, y, _, _ = next(it)
                all_x.append(x)
                all_y.append(y)

            x = torch.cat(all_x, dim=0).to(cfg.device)
            y = torch.cat(all_y, dim=0).float().to(cfg.device)

            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accs = [evaluate(model, loader, cfg.device)["acc_true"] for loader in train_loaders]
        eval_results = evaluate_all(model, eval_loaders, cfg.device)
        print(
            format_eval_results(
                f"[ERM][Epoch {epoch + 1:02d}] Train true accs: {[round(a, 4) for a in train_accs]}",
                eval_results,
            )
        )

    return model


def irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def train_irm(
    model: nn.Module,
    train_loaders,
    eval_loaders: Dict[str, DataLoader],
    cfg: Config,
):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    global_step = 0

    print("\n===== Training IRM =====")
    for epoch in range(cfg.num_epochs):
        model.train()
        env_iters = [iter(loader) for loader in train_loaders]
        steps_per_epoch = min(len(loader) for loader in train_loaders)

        for _ in range(steps_per_epoch):
            env_losses = []
            env_penalties = []

            for it in env_iters:
                x, y, _, _ = next(it)
                x = x.to(cfg.device)
                y = y.float().to(cfg.device)

                logits = model(x)
                env_losses.append(F.binary_cross_entropy_with_logits(logits, y))
                env_penalties.append(irm_penalty(logits, y))

            loss_mean = torch.stack(env_losses).mean()
            penalty_mean = torch.stack(env_penalties).mean()
            penalty_weight = (
                cfg.irm_penalty_weight
                if global_step >= cfg.irm_penalty_anneal_iters
                else 1.0
            )

            loss = loss_mean + penalty_weight * penalty_mean
            if penalty_weight > 1.0:
                loss /= penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        train_accs = [evaluate(model, loader, cfg.device)["acc_true"] for loader in train_loaders]
        eval_results = evaluate_all(model, eval_loaders, cfg.device)
        print(
            format_eval_results(
                f"[IRM][Epoch {epoch + 1:02d}] Train true accs: {[round(a, 4) for a in train_accs]}",
                eval_results,
            )
        )

    return model


def build_loader(dataset: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def print_final_summary(name: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"\n===== {name} Final Summary =====")
    for env_name in ("id", "ood", "intervention"):
        m = metrics[env_name]
        print(
            f"{env_name.upper()}: "
            f"true_acc={m['acc_true']:.4f}, "
            f"noisy_acc={m['acc_noisy']:.4f}, "
            f"color_flip_change={m['color_sensitivity']:.4f}"
        )


def main() -> None:
    set_seed(42)
    cfg = Config()

    print("Using device:", cfg.device)
    print("Building Colored MNIST environments...")
    print(
        f"Train envs={cfg.train_env_probs}, "
        f"ID={cfg.id_env_prob}, OOD={cfg.ood_env_prob}, "
        f"Intervention={cfg.intervention_env_prob}, "
        f"label_noise={cfg.label_flip_prob}"
    )

    train_envs, eval_envs = make_env_datasets(cfg)
    train_loaders = [build_loader(ds, cfg, shuffle=True) for ds in train_envs]
    eval_loaders = {
        name: build_loader(ds, cfg, shuffle=False) for name, ds in eval_envs.items()
    }

    erm_model = SmallCNN(hidden_dim=cfg.hidden_dim)
    train_erm(erm_model, train_loaders, eval_loaders, cfg)
    erm_metrics = evaluate_all(erm_model, eval_loaders, cfg.device)

    irm_model = SmallCNN(hidden_dim=cfg.hidden_dim)
    train_irm(irm_model, train_loaders, eval_loaders, cfg)
    irm_metrics = evaluate_all(irm_model, eval_loaders, cfg.device)

    print_final_summary("ERM", erm_metrics)
    print_final_summary("IRM", irm_metrics)

    print("\nInterpretation:")
    print("1. ID evaluation measures performance in environments with similar correlations.")
    print("2. OOD evaluation checks whether the model still relies on digit shape after color correlation is reversed.")
    print("3. Intervention evaluation uses an approximately independent color environment to test robustness to color interventions.")
    print("4. Lower color_flip_change indicates that predictions depend less on the color channel.")
    print("5. This experiment can support the claim that the model learns more stable cross-environment features.")
    print("6. However, it still cannot by itself prove general causal discovery or causal inference capability.")


if __name__ == "__main__":
    main()
