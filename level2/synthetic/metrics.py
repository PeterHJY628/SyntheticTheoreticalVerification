"""Population-level metrics: Brier risk, Jacobian diagnostics."""

from __future__ import annotations

import torch


def softmax_jacobian_frobenius(probabilities: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of J(p) = diag(p) - pp^T for each row."""
    outer = probabilities.unsqueeze(2) * probabilities.unsqueeze(1)
    jacobian = torch.diag_embed(probabilities) - outer
    return torch.linalg.norm(jacobian, ord="fro", dim=(1, 2))


@torch.inference_mode()
def mean_jacobian_norm(
    probabilities: torch.Tensor,
    batch_size: int = 65536,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(probabilities), batch_size):
        end = min(start + batch_size, len(probabilities))
        norms = softmax_jacobian_frobenius(probabilities[start:end])
        total += norms.sum().item()
        count += norms.numel()
    return total / count


@torch.inference_mode()
def mean_confidence(
    probabilities: torch.Tensor,
    batch_size: int = 65536,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(probabilities), batch_size):
        end = min(start + batch_size, len(probabilities))
        conf = probabilities[start:end].max(dim=1).values
        total += conf.sum().item()
        count += conf.numel()
    return total / count


def brier_risk_hard(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Population Brier risk with one-hot labels."""
    one_hot = torch.zeros_like(predictions)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return float((predictions - one_hot).square().sum(dim=1).mean().item())


@torch.inference_mode()
def population_brier_risk(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 65536,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(predictions), batch_size):
        end = min(start + batch_size, len(predictions))
        one_hot = torch.zeros(end - start, predictions.shape[1], device=predictions.device)
        one_hot.scatter_(1, labels[start:end].unsqueeze(1), 1.0)
        loss = (predictions[start:end] - one_hot).square().sum(dim=1)
        total += loss.sum().item()
        count += loss.numel()
    return total / count


@torch.inference_mode()
def population_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 65536,
) -> float:
    correct = 0
    count = 0
    for start in range(0, len(predictions), batch_size):
        end = min(start + batch_size, len(predictions))
        pred_labels = predictions[start:end].argmax(dim=1)
        correct += (pred_labels == labels[start:end]).sum().item()
        count += end - start
    return correct / count
