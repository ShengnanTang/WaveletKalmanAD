"""Small PyTorch helpers shared by detector implementations."""

from __future__ import annotations

import copy
from typing import Optional

import torch


def get_gpu(cuda: bool = True) -> torch.device:
    """Return the preferred torch device."""
    return torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, epoch: int, lradj: str, learning_rate: float) -> None:
    """Apply a simple learning-rate schedule compatible with legacy callers."""
    if lradj == "consistent":
        lr = learning_rate
    elif lradj in {"type1", "halving"}:
        lr = learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif lradj in {"type2", "step"}:
        lr = learning_rate * (0.5 ** ((epoch - 1) // 5))
    else:
        lr = learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class EarlyStoppingTorch:
    """Early stopping with optional in-memory best-state restore."""

    def __init__(self, path: Optional[str] = None, patience: int = 3, delta: float = 0.0):
        self.path = path
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            if self.path:
                torch.save(self.best_state, self.path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)

