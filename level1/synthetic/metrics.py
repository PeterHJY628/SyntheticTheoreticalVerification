"""Population-level metrics for theorem validation."""

from __future__ import annotations

import torch


def uncentered_second_moment(x: torch.Tensor) -> torch.Tensor:
    return x.T @ x / x.shape[0]


def moment_shift_metrics(
    x_source: torch.Tensor,
    x_target: torch.Tensor,
) -> dict[str, float | torch.Tensor]:
    m_source = uncentered_second_moment(x_source)
    m_target = uncentered_second_moment(x_target)
    delta_m = m_target - m_source
    return {
        "m_source": m_source,
        "m_target": m_target,
        "delta_m": delta_m,
        "moment_shift_fro": float(torch.linalg.matrix_norm(delta_m, ord="fro").item()),
        "source_mean_norm": float(x_source.mean(dim=0).norm().item()),
        "target_mean_norm": float(x_target.mean(dim=0).norm().item()),
    }


def bayes_brier_risk(probabilities: torch.Tensor) -> torch.Tensor:
    pointwise = 1.0 - probabilities.square().sum(dim=1)
    return pointwise.mean()


def brier_risk_metrics(
    p_source: torch.Tensor,
    p_target: torch.Tensor,
) -> dict[str, float]:
    risk_source = bayes_brier_risk(p_source)
    risk_target = bayes_brier_risk(p_target)
    return {
        "source_brier_risk": float(risk_source.item()),
        "target_brier_risk": float(risk_target.item()),
        "risk_gap": float((risk_target - risk_source).item()),
    }


def pointwise_true_brier(
    prediction: torch.Tensor,
    true_probability: torch.Tensor,
) -> torch.Tensor:
    """E_{Y|x}[||h(x) - Y||^2] = 1 + ||h||^2 - 2 h^T p*."""
    return (
        1.0
        + prediction.square().sum(dim=1)
        - 2.0 * (prediction * true_probability).sum(dim=1)
    )


def population_true_brier_risk(
    prediction: torch.Tensor,
    true_probability: torch.Tensor,
) -> float:
    return float(pointwise_true_brier(prediction, true_probability).mean().item())


def moment_gap_fro(x_l: torch.Tensor, x_u: torch.Tensor) -> float:
    m_l = uncentered_second_moment(x_l)
    m_u = uncentered_second_moment(x_u)
    return float(torch.linalg.matrix_norm(m_u - m_l, ord="fro").item())
