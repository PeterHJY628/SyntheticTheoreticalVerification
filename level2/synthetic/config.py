"""Experiment configuration for Level 2 density-gap spirals."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpiralConfig:
    r_center_min: float = 0.02
    r_center_max: float = 0.25
    r_outer_min: float = 0.60
    r_outer_max: float = 1.00
    perturbation: float = 0.005
    seed: int = 42


@dataclass
class CriticConfig:
    hidden_dim: int = 64
    critic_steps: int = 40_000
    critic_batch_size: int = 4096
    critic_patience: int = 12
    critic_lr: float = 1e-3
    critic_weight_decay: float = 1e-4
    validation_interval: int = 2000
    validation_size: int = 4096
    critic_patience: int = 20
    critic_tolerance: float = 1e-5
    population_batch_size: int = 65536
    seed: int = 42


@dataclass
class ActiveLearningConfig:
    n_pool: int = 200_000
    n_eval_center: int = 200_000
    n_eval_outer: int = 200_000
    initial_labeled: int = 10
    query_size: int = 10
    n_episodes: int = 20
    feature_dim: int = 2
    num_classes: int = 2
    classifier_max_steps: int = 5000
    classifier_lr: float = 0.01
    classifier_stale_patience: int = 200
    classifier_tol: float = 1e-8
    t_soft: float = 5.0
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    spiral: SpiralConfig = field(default_factory=SpiralConfig)
    al: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    density_gap_r_mins: list[float] = field(
        default_factory=lambda: [0.35, 0.45, 0.55, 0.65, 0.75]
    )
    temperature_values: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 5.0, 10.0]
    )
