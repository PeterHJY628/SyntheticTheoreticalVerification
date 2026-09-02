"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass

# Softmax temperature for IPM critic inputs (logits / T); matches dino_core.
IPM_PROB_TEMPERATURE = 2.0

CRITIC_SPECTRAL = "spectral"
CRITIC_BILINEAR = "bilinear"
CRITIC_TYPES = (CRITIC_SPECTRAL, CRITIC_BILINEAR, "both")


@dataclass
class PopulationConfig:
    n_population: int = 1_000_000
    rotation_deg: float = 60.0
    scale_x: float = 4.0
    scale_y: float = 0.25
    beta: float = 4.0
    l2_normalize: bool = True
    seed: int = 42


@dataclass
class CriticConfig:
    critic_type: str = CRITIC_SPECTRAL
    acquisition_critic: str = CRITIC_SPECTRAL
    hidden_dim: int = 64
    bilinear_rank: int = 32
    critic_steps: int = 1_000_000
    critic_batch_size: int = 1024
    critic_lr: float = 1e-3
    critic_weight_decay: float = 1e-4
    validation_interval: int = 5000
    validation_size: int = 4096
    critic_patience: int = 20
    critic_tolerance: float = 1e-5
    population_batch_size: int = 65536
    seed: int = 42


@dataclass
class ActiveLearningConfig:
    n_pool: int = 1_000_000
    n_test: int = 1_000_000
    initial_labeled: int = 10
    initial_sample_buffer: int = 100
    query_size: int = 10
    n_episodes: int = 20
    proxy_chunk_size: int = 4096
    feature_dim: int = 2
    num_classes: int = 2
    classifier_max_steps: int = 2000
    classifier_lr: float = 0.03
    classifier_stale_patience: int = 100
    classifier_tol: float = 1e-8
    seed: int = 42
    device: str = "cuda"
