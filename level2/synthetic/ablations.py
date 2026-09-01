"""Ablation studies for Level 2."""

from __future__ import annotations

import torch

from synthetic.classifier import fit_classifier, logits_batched, temperature_softmax
from synthetic.config import ActiveLearningConfig, CriticConfig, SpiralConfig
from synthetic.critic import train_and_evaluate_critic
from synthetic.metrics import (
    mean_confidence,
    mean_jacobian_norm,
    population_accuracy,
    population_brier_risk,
)
from synthetic.population import (
    generate_active_learning_data,
    generate_outer_eval_for_r_min,
    sample_class_balanced_center,
    sample_spiral_region,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_density_gap_ablation(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    r_mins: list[float],
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Train on initial center labels; sweep outer-tail minimum radius."""
    device = _resolve_device(al_cfg.device)
    spiral_cfg = SpiralConfig(
        r_center_min=spiral_cfg.r_center_min,
        r_center_max=spiral_cfg.r_center_max,
        r_outer_min=spiral_cfg.r_outer_min,
        r_outer_max=spiral_cfg.r_outer_max,
        perturbation=spiral_cfg.perturbation,
        seed=seed,
    )

    x_l, y_l = sample_class_balanced_center(
        al_cfg.initial_labeled // 2,
        spiral_cfg,
        seed=seed + 100,
    )
    x_l = x_l.to(device)
    y_l = y_l.to(device)

    x_eval_center, y_eval_center, _ = sample_spiral_region(
        al_cfg.n_eval_center,
        spiral_cfg.r_center_min,
        spiral_cfg.r_center_max,
        spiral_cfg.perturbation,
        seed=seed + 2,
        balanced=True,
    )
    x_eval_center = x_eval_center.to(device)
    y_eval_center = y_eval_center.to(device)

    classifier = fit_classifier(
        x_l,
        y_l,
        feature_dim=al_cfg.feature_dim,
        num_classes=al_cfg.num_classes,
        seed=seed,
        max_steps=al_cfg.classifier_max_steps,
        lr=al_cfg.classifier_lr,
        stale_patience=al_cfg.classifier_stale_patience,
        tol=al_cfg.classifier_tol,
    )

    batch = critic_cfg.population_batch_size
    p_eval_center = classifier(x_eval_center)
    risk_center = population_brier_risk(p_eval_center, y_eval_center, batch)

    rows: list[dict] = []
    for r_min in r_mins:
        x_outer, y_outer = generate_outer_eval_for_r_min(
            al_cfg.n_eval_outer,
            r_min,
            spiral_cfg.r_outer_max,
            spiral_cfg.perturbation,
            seed=seed + int(r_min * 1000),
        )
        x_outer = x_outer.to(device)
        y_outer = y_outer.to(device)

        logits_outer = logits_batched(classifier, x_outer, batch)
        p_raw = temperature_softmax(logits_outer, 1.0)
        p_soft = temperature_softmax(logits_outer, al_cfg.t_soft)
        p_eval_outer = classifier(x_outer)

        risk_outer = population_brier_risk(p_eval_outer, y_outer, batch)
        true_risk_gap = risk_outer - risk_center

        logits_l = logits_batched(classifier, x_l, batch)
        p_l_raw = temperature_softmax(logits_l, 1.0)
        p_l_soft = temperature_softmax(logits_l, al_cfg.t_soft)
        p_u_raw = p_raw
        p_u_soft = p_soft

        input_dim = al_cfg.feature_dim + al_cfg.num_classes
        _, ipm_raw = train_and_evaluate_critic(
            x_l, p_l_raw, x_outer, p_u_raw, critic_cfg, input_dim, seed=seed
        )
        _, ipm_soft = train_and_evaluate_critic(
            x_l, p_l_soft, x_outer, p_u_soft, critic_cfg, input_dim, seed=seed + 1
        )

        row = {
            "r_min_outer": r_min,
            "risk_center": risk_center,
            "risk_outer": risk_outer,
            "true_risk_gap": true_risk_gap,
            "accuracy_outer": population_accuracy(p_eval_outer, y_outer, batch),
            "mean_confidence_outer": mean_confidence(p_raw, batch),
            "jacobian_raw": mean_jacobian_norm(p_raw, batch),
            "jacobian_soft": mean_jacobian_norm(p_soft, batch),
            "ipm_raw": ipm_raw,
            "ipm_soft": ipm_soft,
            "bound_margin_raw": ipm_raw - true_risk_gap,
            "bound_margin_soft": ipm_soft - true_risk_gap,
            "seed": seed,
        }
        rows.append(row)
        if verbose:
            print(
                f"  r_min={r_min:.2f}: gap={true_risk_gap:.4f}, "
                f"ipm_raw={ipm_raw:.4f}, ipm_soft={ipm_soft:.4f}",
                flush=True,
            )
    return rows


def run_temperature_ablation(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    temperatures: list[float],
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """At episode 0, sweep temperature for Jacobian/IPM/bound margin."""
    device = _resolve_device(al_cfg.device)
    data = generate_active_learning_data(
        SpiralConfig(
            r_center_min=spiral_cfg.r_center_min,
            r_center_max=spiral_cfg.r_center_max,
            r_outer_min=spiral_cfg.r_outer_min,
            r_outer_max=spiral_cfg.r_outer_max,
            perturbation=spiral_cfg.perturbation,
            seed=seed,
        ),
        al_cfg,
    )

    x_l = data["x_initial"].to(device)
    y_l = data["y_initial"].to(device)
    x_u = data["x_pool"].to(device)
    x_eval_center = data["x_eval_center"].to(device)
    y_eval_center = data["y_eval_center"].to(device)
    x_eval_outer = data["x_eval_outer"].to(device)
    y_eval_outer = data["y_eval_outer"].to(device)

    classifier = fit_classifier(
        x_l,
        y_l,
        feature_dim=al_cfg.feature_dim,
        num_classes=al_cfg.num_classes,
        seed=seed,
        max_steps=al_cfg.classifier_max_steps,
        lr=al_cfg.classifier_lr,
        stale_patience=al_cfg.classifier_stale_patience,
        tol=al_cfg.classifier_tol,
    )

    batch = critic_cfg.population_batch_size
    p_eval_center = classifier(x_eval_center)
    p_eval_outer = classifier(x_eval_outer)
    risk_center = population_brier_risk(p_eval_center, y_eval_center, batch)
    risk_outer = population_brier_risk(p_eval_outer, y_eval_outer, batch)
    true_risk_gap = risk_outer - risk_center

    logits_l = logits_batched(classifier, x_l, batch)
    logits_u = logits_batched(classifier, x_u, batch)
    logits_outer = logits_batched(classifier, x_eval_outer, batch)
    input_dim = al_cfg.feature_dim + al_cfg.num_classes

    rows: list[dict] = []
    for temperature in temperatures:
        p_l = temperature_softmax(logits_l, temperature)
        p_u = temperature_softmax(logits_u, temperature)
        p_outer = temperature_softmax(logits_outer, temperature)

        _, ipm = train_and_evaluate_critic(
            x_l, p_l, x_u, p_u, critic_cfg, input_dim, seed=seed + int(temperature * 10)
        )

        row = {
            "temperature": temperature,
            "true_risk_gap": true_risk_gap,
            "mean_confidence_outer": mean_confidence(p_outer, batch),
            "jacobian_norm": mean_jacobian_norm(p_outer, batch),
            "ipm": ipm,
            "bound_margin": ipm - true_risk_gap,
            "seed": seed,
        }
        rows.append(row)
        if verbose:
            print(
                f"  T={temperature:.1f}: jac={row['jacobian_norm']:.6f}, "
                f"ipm={ipm:.4f}, margin={row['bound_margin']:.4f}",
                flush=True,
            )
    return rows
