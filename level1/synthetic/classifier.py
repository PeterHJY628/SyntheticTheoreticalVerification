"""Linear Brier classifier aligned with simplified_dino_core conventions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from synthetic.config import IPM_PROB_TEMPERATURE


class LinearClassifier(nn.Module):
    """Linear head; forward returns raw logits (softmax applied outside)."""

    def __init__(self, feature_dim: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes, bias=False)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


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

    previous = float("inf")
    stale = 0
    for _ in range(max_steps):
        model.train()
        logits = model(x_l)
        predictions = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(y_l, num_classes=num_classes).float()
        loss = F.mse_loss(predictions, targets_one_hot, reduction="sum")
        loss = loss / x_l.size(0)

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
def logits_batched(
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
def predict_batched(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 65536,
) -> torch.Tensor:
    """Standard T=1 softmax probabilities for risk / accuracy."""
    return F.softmax(logits_batched(model, x, batch_size), dim=1)


@torch.inference_mode()
def ipm_probabilities_batched(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 65536,
    temperature: float = IPM_PROB_TEMPERATURE,
) -> torch.Tensor:
    """Temperature-softened probs for IPM critic (detached)."""
    logits = logits_batched(model, x, batch_size)
    return F.softmax(logits / temperature, dim=1).detach()
