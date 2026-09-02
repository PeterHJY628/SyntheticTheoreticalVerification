"""IPM critics for the synthetic experiment (spectral-norm MLP and compact bilinear)."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from synthetic.config import (
    CRITIC_BILINEAR,
    CRITIC_SPECTRAL,
    CriticConfig,
)

_MAX_TRAIN_LOG_POINTS = 5_000

_CRITIC_SEED_OFFSET = {
    CRITIC_SPECTRAL: 0,
    CRITIC_BILINEAR: 10_000,
}


class CompactBilinearNetwork(nn.Module):
    """Bilinear critic: x^T W_cross y_hat + x^T A x with A = L L^T (low-rank)."""

    name = CRITIC_BILINEAR

    def __init__(self, feature_dim: int, num_classes: int, rank: int = 32) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.rank = rank
        self.W_cross = nn.Parameter(torch.empty(feature_dim, num_classes))
        self.L = nn.Parameter(torch.empty(feature_dim, rank))
        nn.init.xavier_uniform_(self.W_cross)
        nn.init.xavier_uniform_(self.L)

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        cross_term = (torch.matmul(x, self.W_cross) * y_hat).sum(dim=1)
        auto_term = (torch.matmul(x, self.L) ** 2).sum(dim=1)
        return cross_term + auto_term


class SpectralNormLayerIPM(nn.Module):
    """1-spectral-norm-layer IPM with configurable hidden width."""

    name = CRITIC_SPECTRAL

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        hidden = spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.net = nn.Sequential(
            hidden,
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([features, probabilities], dim=1)).squeeze(1)


def resolve_critic_types(critic_type: str) -> list[str]:
    """Expand ``both`` into the two trainable critic variants."""
    if critic_type == "both":
        return [CRITIC_SPECTRAL, CRITIC_BILINEAR]
    if critic_type not in (CRITIC_SPECTRAL, CRITIC_BILINEAR):
        raise ValueError(
            f"critic_type must be one of spectral, bilinear, both; got {critic_type!r}"
        )
    return [critic_type]


def normalize_critic_config(config: CriticConfig) -> CriticConfig:
    """Align acquisition critic with a single trained variant; validate ``both`` mode."""
    types = resolve_critic_types(config.critic_type)
    if len(types) == 1:
        return replace(config, acquisition_critic=types[0])
    if config.acquisition_critic not in types:
        raise ValueError(
            f"acquisition_critic={config.acquisition_critic!r} must be one of {types}"
        )
    return config


def build_critic(
    critic_type: str,
    feature_dim: int,
    num_classes: int,
    config: CriticConfig,
    device: torch.device,
) -> nn.Module:
    """Instantiate a critic module (shared forward: features, probabilities)."""
    if critic_type == CRITIC_SPECTRAL:
        input_dim = feature_dim + num_classes
        return SpectralNormLayerIPM(input_dim, config.hidden_dim).to(device)
    if critic_type == CRITIC_BILINEAR:
        return CompactBilinearNetwork(
            feature_dim,
            num_classes,
            rank=config.bilinear_rank,
        ).to(device)
    raise ValueError(f"unknown critic_type: {critic_type!r}")


def critic_training_seed(base_seed: int, episode: int, critic_type: str) -> int:
    offset = _CRITIC_SEED_OFFSET.get(critic_type, 0)
    return base_seed + episode + offset


def _sample_from_pool(pool: torch.Tensor, size: int) -> torch.Tensor:
    choice = torch.randint(len(pool), (size,), device=pool.device)
    return pool[choice]


def _subsample_series(
    values: list[float],
    max_points: int = _MAX_TRAIN_LOG_POINTS,
) -> tuple[list[float], list[int]]:
    """Downsample long per-step series for JSON/plot storage."""
    if len(values) <= max_points:
        return values, list(range(len(values)))
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    unique_indices = sorted(set(indices.tolist()))
    return [values[i] for i in unique_indices], unique_indices


def _build_training_trace(
    *,
    train_loss: list[float],
    train_objective: list[float],
    validation_objective: list[float],
    validation_steps: list[int],
    critic_steps: int,
    best_validation_objective: float,
    stopped_early: bool,
    config: CriticConfig,
) -> dict[str, Any]:
    """Package critic training curves and convergence diagnostics."""
    train_loss_ds, train_steps_ds = _subsample_series(train_loss)
    train_objective_ds, _ = _subsample_series(train_objective)

    converged = stopped_early
    if validation_objective:
        tail = validation_objective[-config.critic_patience :]
        if len(tail) >= config.critic_patience:
            improvements = [
                tail[i] - tail[i - 1] for i in range(1, len(tail))
            ]
            plateau = all(
                improvement <= config.critic_tolerance for improvement in improvements
            )
            converged = converged or plateau

    return {
        "train_loss": train_loss_ds,
        "train_steps": train_steps_ds,
        "train_objective": train_objective_ds,
        "validation_objective": validation_objective,
        "validation_steps": validation_steps,
        "critic_steps": critic_steps,
        "best_validation_objective": best_validation_objective,
        "stopped_early": stopped_early,
        "converged": converged,
        "train_points_logged": len(train_loss),
        "validation_checks": len(validation_objective),
    }


def train_ipm_critic(
    critic: nn.Module,
    x_l: torch.Tensor,
    p_l: torch.Tensor,
    x_u: torch.Tensor,
    p_u: torch.Tensor,
    config: CriticConfig,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train critic to maximize E_U[f] - E_L[f] with early stopping."""
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_lr,
        weight_decay=config.critic_weight_decay,
    )

    critic.train()
    best_objective = -float("inf")
    best_state = None
    stale = 0
    steps_run = 0
    stopped_early = False

    train_loss: list[float] = []
    train_objective: list[float] = []
    validation_objective: list[float] = []
    validation_steps: list[int] = []

    for step in range(config.critic_steps):
        steps_run = step + 1
        critic.train()
        l_idx = torch.randint(0, len(x_l), (config.critic_batch_size,), device=x_l.device)
        u_idx = torch.randint(0, len(x_u), (config.critic_batch_size,), device=x_u.device)

        objective = critic(x_u[u_idx], p_u[u_idx]).mean() - critic(x_l[l_idx], p_l[l_idx]).mean()
        loss = -objective
        train_objective.append(float(objective.detach().item()))
        train_loss.append(float(loss.detach().item()))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps_run % config.validation_interval == 0:
            critic.eval()
            with torch.inference_mode():
                vl = torch.randint(
                    0,
                    len(x_l),
                    (min(config.validation_size, len(x_l)),),
                    device=x_l.device,
                )
                vu = torch.randint(
                    0,
                    len(x_u),
                    (config.validation_size,),
                    device=x_u.device,
                )
                val_objective = float(
                    critic(x_u[vu], p_u[vu]).mean() - critic(x_l[vl], p_l[vl]).mean()
                )
            critic.train()
            validation_objective.append(val_objective)
            validation_steps.append(steps_run)

            if val_objective > best_objective + config.critic_tolerance:
                best_objective = val_objective
                best_state = copy.deepcopy(critic.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= config.critic_patience:
                stopped_early = True
                break

    if best_state is not None:
        critic.load_state_dict(best_state)
    critic.eval()

    trace = _build_training_trace(
        train_loss=train_loss,
        train_objective=train_objective,
        validation_objective=validation_objective,
        validation_steps=validation_steps,
        critic_steps=steps_run,
        best_validation_objective=best_objective,
        stopped_early=stopped_early,
        config=config,
    )
    return critic, trace


def train_and_evaluate_critic(
    critic_type: str,
    x_l: torch.Tensor,
    p_l: torch.Tensor,
    x_u: torch.Tensor,
    p_u: torch.Tensor,
    critic_cfg: CriticConfig,
    feature_dim: int,
    num_classes: int,
    episode: int,
    batch_size: int,
) -> tuple[nn.Module, dict[str, Any], float]:
    """Build, train, and population-evaluate one critic variant."""
    device = x_l.device
    critic = build_critic(critic_type, feature_dim, num_classes, critic_cfg, device)
    torch.manual_seed(critic_training_seed(critic_cfg.seed, episode, critic_type))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(critic_training_seed(critic_cfg.seed, episode, critic_type))
    critic, trace = train_ipm_critic(critic, x_l, p_l, x_u, p_u, critic_cfg)
    trace["critic_type"] = critic_type
    ipm_value = evaluate_ipm(critic, x_l, p_l, x_u, p_u, batch_size)
    return critic, trace, ipm_value


def train_critic(
    critic: nn.Module,
    features: torch.Tensor,
    probabilities: torch.Tensor,
    labeled_idx: torch.Tensor,
    unlabeled_idx: torch.Tensor,
    config: CriticConfig,
) -> dict[str, Any]:
    """Index-based critic training for static population sweep experiments."""
    x_l = features[labeled_idx]
    p_l = probabilities[labeled_idx]
    x_u = features[unlabeled_idx]
    p_u = probabilities[unlabeled_idx]
    _, trace = train_ipm_critic(critic, x_l, p_l, x_u, p_u, config)
    return trace


@torch.inference_mode()
def critic_mean(
    critic: nn.Module,
    x: torch.Tensor,
    p: torch.Tensor,
    batch_size: int,
) -> float:
    critic.eval()
    total = 0.0
    count = 0
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        score = critic(x[start:end], p[start:end])
        total += score.sum().item()
        count += score.numel()
    return total / count


def evaluate_ipm(
    critic: nn.Module,
    x_l: torch.Tensor,
    p_l: torch.Tensor,
    x_u: torch.Tensor,
    p_u: torch.Tensor,
    batch_size: int,
) -> float:
    mu_u = critic_mean(critic, x_u, p_u, batch_size)
    mu_l = critic_mean(critic, x_l, p_l, batch_size)
    return mu_u - mu_l


def estimate_population_ipm(
    critic: nn.Module,
    x_source: torch.Tensor,
    p_source: torch.Tensor,
    x_target: torch.Tensor,
    p_target: torch.Tensor,
    config: CriticConfig,
) -> float:
    return evaluate_ipm(
        critic,
        x_source,
        p_source,
        x_target,
        p_target,
        config.population_batch_size,
    )


@torch.inference_mode()
def select_topk_by_critic(
    critic: nn.Module,
    x_u: torch.Tensor,
    p_u: torch.Tensor,
    k: int = 10,
    batch_size: int = 65536,
) -> torch.Tensor:
    """Return local indices into x_u with highest critic scores."""
    critic.eval()
    best_scores: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None

    for start in range(0, len(x_u), batch_size):
        end = min(start + batch_size, len(x_u))
        scores = critic(x_u[start:end], p_u[start:end])
        kk = min(k, len(scores))
        values, idx = torch.topk(scores, k=kk)
        idx = idx + start

        if best_scores is None:
            best_scores = values
            best_indices = idx
        else:
            all_scores = torch.cat([best_scores, values])
            all_indices = torch.cat([best_indices, idx])
            kk = min(k, len(all_scores))
            best_scores, order = torch.topk(all_scores, k=kk)
            best_indices = all_indices[order]

    assert best_indices is not None
    return best_indices
