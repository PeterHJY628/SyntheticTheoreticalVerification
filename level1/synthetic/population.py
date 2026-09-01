"""Procedurally generated source/target populations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from synthetic.config import ActiveLearningConfig, PopulationConfig

GMM_MEANS = torch.tensor(
    [
        [2.4, 0.6],
        [-2.4, -0.6],
        [0.7, 1.6],
        [-0.7, -1.6],
    ],
    dtype=torch.float32,
)

GMM_WEIGHTS = torch.tensor([0.30, 0.30, 0.20, 0.20], dtype=torch.float32)

GMM_COVS = torch.tensor(
    [
        [[0.35, 0.08], [0.08, 0.15]],
        [[0.35, 0.08], [0.08, 0.15]],
        [[0.18, -0.03], [-0.03, 0.28]],
        [[0.18, -0.03], [-0.03, 0.28]],
    ],
    dtype=torch.float32,
)


def sample_gmm(n: int, seed: int) -> torch.Tensor:
    """Sample from a centrally symmetric 2D GMM using antithetic pairs."""
    if n % 2 != 0:
        raise ValueError("n must be even for antithetic sampling")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    half = n // 2
    chol = torch.linalg.cholesky(GMM_COVS)

    component = torch.multinomial(
        GMM_WEIGHTS,
        half,
        replacement=True,
        generator=generator,
    )
    eps = torch.randn(half, 2, generator=generator)
    noise = torch.bmm(chol[component], eps.unsqueeze(-1)).squeeze(-1)
    x_half = GMM_MEANS[component] + noise
    return torch.cat([x_half, -x_half], dim=0)


def transformation_matrix(
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:
    theta = math.radians(rotation_deg)
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=torch.float32,
    )
    scaling = torch.tensor(
        [
            [scale_x, 0.0],
            [0.0, scale_y],
        ],
        dtype=torch.float32,
    )
    return rotation @ scaling


def build_w_star(beta: float) -> torch.Tensor:
    w = torch.tensor([1.0, -0.65], dtype=torch.float32)
    w = w / torch.linalg.vector_norm(w)
    return torch.stack([-0.5 * beta * w, 0.5 * beta * w])


def true_probabilities(x: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x @ w_star.T, dim=1)


def sample_labels(probabilities: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    uniform = torch.rand(probabilities.shape[0], generator=generator)
    return (uniform < probabilities[:, 1]).long()


def generate_active_learning_data(
    cfg: PopulationConfig,
    al_cfg: ActiveLearningConfig,
) -> dict[str, torch.Tensor]:
    """Build initial source labels, target pool, and independent test population."""
    x_initial_raw = sample_gmm(al_cfg.initial_sample_buffer, seed=cfg.seed)[
        : al_cfg.initial_labeled
    ]
    x_pool_base = sample_gmm(al_cfg.n_pool, seed=cfg.seed + 1)
    x_test_base = sample_gmm(al_cfg.n_test, seed=cfg.seed + 2)

    transform = transformation_matrix(cfg.rotation_deg, cfg.scale_x, cfg.scale_y)
    x_pool_raw = x_pool_base @ transform.T
    x_test_raw = x_test_base @ transform.T

    if cfg.l2_normalize:
        x_initial = F.normalize(x_initial_raw, p=2, dim=1)
        x_pool = F.normalize(x_pool_raw, p=2, dim=1)
        x_test = F.normalize(x_test_raw, p=2, dim=1)
    else:
        x_initial = x_initial_raw
        x_pool = x_pool_raw
        x_test = x_test_raw

    w_star = build_w_star(cfg.beta)
    return {
        "x_initial": x_initial,
        "x_pool": x_pool,
        "x_test": x_test,
        "p_initial_true": true_probabilities(x_initial, w_star),
        "p_pool_true": true_probabilities(x_pool, w_star),
        "p_test_true": true_probabilities(x_test, w_star),
        "y_initial": sample_labels(true_probabilities(x_initial, w_star), cfg.seed + 10),
        "y_pool_hidden": sample_labels(true_probabilities(x_pool, w_star), cfg.seed + 11),
        "y_test": sample_labels(true_probabilities(x_test, w_star), cfg.seed + 12),
        "w_star": w_star,
        "transform": transform,
    }


def generate_populations(cfg: PopulationConfig) -> dict[str, torch.Tensor]:
    n = cfg.n_population
    x_source_raw = sample_gmm(n, seed=cfg.seed)
    x_target_base = sample_gmm(n, seed=cfg.seed + 1)

    transform = transformation_matrix(cfg.rotation_deg, cfg.scale_x, cfg.scale_y)
    x_target_raw = x_target_base @ transform.T

    if cfg.l2_normalize:
        x_source = F.normalize(x_source_raw, p=2, dim=1)
        x_target = F.normalize(x_target_raw, p=2, dim=1)
    else:
        x_source = x_source_raw
        x_target = x_target_raw

    w_star = build_w_star(cfg.beta)
    p_source = true_probabilities(x_source, w_star)
    p_target = true_probabilities(x_target, w_star)

    return {
        "x_source": x_source,
        "x_target": x_target,
        "p_source": p_source,
        "p_target": p_target,
        "w_star": w_star,
        "transform": transform,
    }
