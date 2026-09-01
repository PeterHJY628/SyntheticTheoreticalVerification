"""Linear Brier classifier for the synthetic active-learning experiment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LinearClassifier(nn.Module):
    """h(x) = softmax(Wx), matching the theorem hypothesis class."""

    def __init__(self, feature_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.linear(x), dim=1)


def fit_classifier(
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    feature_dim: int = 2,
    num_classes: int = 2,
    seed: int = 0,
    max_steps: int = 2000,
    lr: float = 0.03,
    stale_patience: int = 100,
    tol: float = 1e-8,
) -> LinearClassifier:
    torch.manual_seed(seed)
    model = LinearClassifier(feature_dim, num_classes).to(x_l.device)
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
