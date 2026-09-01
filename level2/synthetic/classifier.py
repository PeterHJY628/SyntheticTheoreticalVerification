"""MLP Brier classifier for interleaved spirals."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPClassifier(nn.Module):
    """Nonlinear classifier: 2 -> 64 -> ReLU -> 64 -> ReLU -> 2."""

    def __init__(self, feature_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(x), dim=1)


def fit_classifier(
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    feature_dim: int = 2,
    num_classes: int = 2,
    seed: int = 0,
    max_steps: int = 5000,
    lr: float = 0.01,
    stale_patience: int = 200,
    tol: float = 1e-8,
) -> MLPClassifier:
    torch.manual_seed(seed)
    model = MLPClassifier(feature_dim, num_classes).to(x_l.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    targets = F.one_hot(y_l, num_classes=num_classes).float()

    previous = float("inf")
    stale = 0
    for _ in range(max_steps):
        model.train()
        predictions = model(x_l)
        loss = (predictions - targets).square().sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current = float(loss.detach())
        if abs(previous - current) < tol:
            stale += 1
        else:
            stale = 0
        previous = current
        if stale >= stale_patience:
            break

    model.eval()
    return model


@torch.inference_mode()
def predict_batched(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 65536,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        outputs.append(model(x[start:end]))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def logits_batched(
    model: MLPClassifier,
    x: torch.Tensor,
    batch_size: int = 65536,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        outputs.append(model.forward_logits(x[start:end]))
    return torch.cat(outputs, dim=0)


def temperature_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return F.softmax(logits, dim=1)
    return F.softmax(logits / temperature, dim=1)
