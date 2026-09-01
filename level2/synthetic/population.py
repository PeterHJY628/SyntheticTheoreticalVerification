"""2D interleaved spiral populations with strict support split."""

from __future__ import annotations

import math

import torch

from synthetic.config import ActiveLearningConfig, SpiralConfig


def _spiral_coords(r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Map radius/class to 2D spiral coordinates."""
    theta = 4.0 * math.pi * r + math.pi * y
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def _add_perturbation(
    x: torch.Tensor,
    r: torch.Tensor,
    perturbation: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if perturbation <= 0.0:
        return x
    delta_r = (torch.rand(len(r), generator=generator) * 2.0 - 1.0) * perturbation
    delta_theta = (torch.rand(len(r), generator=generator) * 2.0 - 1.0) * perturbation
    r_new = (r + delta_r).clamp(min=0.0)
    theta = torch.atan2(x[:, 1], x[:, 0]) + delta_theta
    return torch.stack([r_new * torch.cos(theta), r_new * torch.sin(theta)], dim=1)


def sample_spiral_region(
    n: int,
    r_min: float,
    r_max: float,
    perturbation: float,
    seed: int,
    balanced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample n points from spirals with r in [r_min, r_max]."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if balanced and n % 2 == 0:
        half = n // 2
        r = torch.cat(
            [
                r_min + (r_max - r_min) * torch.rand(half, generator=generator),
                r_min + (r_max - r_min) * torch.rand(half, generator=generator),
            ]
        )
        y = torch.cat(
            [
                torch.zeros(half, dtype=torch.long),
                torch.ones(half, dtype=torch.long),
            ]
        )
    else:
        r = r_min + (r_max - r_min) * torch.rand(n, generator=generator)
        y = torch.randint(0, 2, (n,), generator=generator)

    x = _spiral_coords(r, y.float())
    x = _add_perturbation(x, r, perturbation, generator)
    return x, y, r


def sample_class_balanced_center(
    n_per_class: int,
    spiral_cfg: SpiralConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample class-balanced center points for initialization."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    r_min = spiral_cfg.r_center_min
    r_max = spiral_cfg.r_center_max
    perturbation = spiral_cfg.perturbation

    samples_x = []
    samples_y = []
    for class_id in (0, 1):
        r = r_min + (r_max - r_min) * torch.rand(n_per_class, generator=generator)
        y = torch.full((n_per_class,), class_id, dtype=torch.long)
        x = _spiral_coords(r, y.float())
        x = _add_perturbation(x, r, perturbation, generator)
        samples_x.append(x)
        samples_y.append(y)

    return torch.cat(samples_x, dim=0), torch.cat(samples_y, dim=0)


def generate_active_learning_data(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
) -> dict[str, torch.Tensor]:
    """Build center labels, outer pool, and fixed evaluation populations."""
    n_init_per_class = al_cfg.initial_labeled // 2
    x_initial, y_initial = sample_class_balanced_center(
        n_init_per_class,
        spiral_cfg,
        seed=spiral_cfg.seed + 100,
    )

    x_pool, y_pool, r_pool = sample_spiral_region(
        al_cfg.n_pool,
        spiral_cfg.r_outer_min,
        spiral_cfg.r_outer_max,
        spiral_cfg.perturbation,
        seed=spiral_cfg.seed + 1,
        balanced=True,
    )
    x_eval_center, y_eval_center, _ = sample_spiral_region(
        al_cfg.n_eval_center,
        spiral_cfg.r_center_min,
        spiral_cfg.r_center_max,
        spiral_cfg.perturbation,
        seed=spiral_cfg.seed + 2,
        balanced=True,
    )
    x_eval_outer, y_eval_outer, _ = sample_spiral_region(
        al_cfg.n_eval_outer,
        spiral_cfg.r_outer_min,
        spiral_cfg.r_outer_max,
        spiral_cfg.perturbation,
        seed=spiral_cfg.seed + 3,
        balanced=True,
    )

    return {
        "x_initial": x_initial,
        "y_initial": y_initial,
        "x_pool": x_pool,
        "y_pool_hidden": y_pool,
        "r_pool": r_pool,
        "x_eval_center": x_eval_center,
        "y_eval_center": y_eval_center,
        "x_eval_outer": x_eval_outer,
        "y_eval_outer": y_eval_outer,
    }


def generate_outer_eval_for_r_min(
    n: int,
    r_min: float,
    r_max: float,
    perturbation: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outer-tail evaluation population with configurable minimum radius."""
    x, y, _ = sample_spiral_region(
        n,
        r_min,
        r_max,
        perturbation,
        seed=seed,
        balanced=True,
    )
    return x, y
