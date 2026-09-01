"""Spectral-norm IPM critic for the synthetic experiment."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from synthetic.config import CriticConfig


class SpectralNormLayerIPM(nn.Module):
    """1-spectral-norm-layer IPM with configurable hidden width."""

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


def _sample_from_pool(pool: torch.Tensor, size: int) -> torch.Tensor:
    choice = torch.randint(len(pool), (size,), device=pool.device)
    return pool[choice]


def train_ipm_critic(
    critic: nn.Module,
    x_l: torch.Tensor,
    p_l: torch.Tensor,
    x_u: torch.Tensor,
    p_u: torch.Tensor,
    config: CriticConfig,
) -> nn.Module:
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

    for step in range(config.critic_steps):
        critic.train()
        l_idx = torch.randint(0, len(x_l), (config.critic_batch_size,), device=x_l.device)
        u_idx = torch.randint(0, len(x_u), (config.critic_batch_size,), device=x_u.device)

        objective = critic(x_u[u_idx], p_u[u_idx]).mean() - critic(x_l[l_idx], p_l[l_idx]).mean()
        loss = -objective

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % config.validation_interval == 0:
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
                validation_objective = float(
                    critic(x_u[vu], p_u[vu]).mean() - critic(x_l[vl], p_l[vl]).mean()
                )
            critic.train()

            if validation_objective > best_objective + config.critic_tolerance:
                best_objective = validation_objective
                best_state = copy.deepcopy(critic.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= config.critic_patience:
                break

    if best_state is not None:
        critic.load_state_dict(best_state)
    critic.eval()
    return critic


def train_critic(
    critic: nn.Module,
    features: torch.Tensor,
    probabilities: torch.Tensor,
    labeled_idx: torch.Tensor,
    unlabeled_idx: torch.Tensor,
    config: CriticConfig,
) -> dict:
    """Index-based critic training for static population sweep experiments."""
    x_l = features[labeled_idx]
    p_l = probabilities[labeled_idx]
    x_u = features[unlabeled_idx]
    p_u = probabilities[unlabeled_idx]
    train_ipm_critic(critic, x_l, p_l, x_u, p_u, config)
    return {"critic_steps": config.critic_steps, "best_validation_objective": None}


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
