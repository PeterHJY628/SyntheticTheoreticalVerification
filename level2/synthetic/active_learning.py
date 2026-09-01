"""Active-learning loop for Level 2 density-gap experiment."""

from __future__ import annotations

import torch

from synthetic.classifier import (
    fit_classifier,
    logits_batched,
    predict_batched,
    temperature_softmax,
)
from synthetic.config import ActiveLearningConfig, CriticConfig, SpiralConfig
from synthetic.critic import select_topk_by_critic, train_and_evaluate_critic
from synthetic.metrics import (
    mean_confidence,
    mean_jacobian_norm,
    population_accuracy,
    population_brier_risk,
)
from synthetic.population import generate_active_learning_data


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _episode_metrics(
    classifier,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    x_u: torch.Tensor,
    x_eval_center: torch.Tensor,
    y_eval_center: torch.Tensor,
    x_eval_outer: torch.Tensor,
    y_eval_outer: torch.Tensor,
    logits_outer: torch.Tensor,
    critic_cfg: CriticConfig,
    al_cfg: ActiveLearningConfig,
    episode: int,
    seed: int,
) -> tuple[dict, torch.nn.Module]:
    batch = critic_cfg.population_batch_size
    p_raw_outer = temperature_softmax(logits_outer, 1.0)
    p_soft_outer = temperature_softmax(logits_outer, al_cfg.t_soft)

    p_eval_center = predict_batched(classifier, x_eval_center, batch)
    p_eval_outer = predict_batched(classifier, x_eval_outer, batch)

    risk_center = population_brier_risk(p_eval_center, y_eval_center, batch)
    risk_outer = population_brier_risk(p_eval_outer, y_eval_outer, batch)
    true_risk_gap = risk_outer - risk_center

    p_l_raw = predict_batched(classifier, x_l, batch)
    p_u_raw = predict_batched(classifier, x_u, batch)
    logits_l = logits_batched(classifier, x_l, batch)
    logits_u = logits_batched(classifier, x_u, batch)
    p_l_soft = temperature_softmax(logits_l, al_cfg.t_soft)
    p_u_soft = temperature_softmax(logits_u, al_cfg.t_soft)

    input_dim = al_cfg.feature_dim + al_cfg.num_classes
    critic_raw, ipm_raw = train_and_evaluate_critic(
        x_l,
        p_l_raw,
        x_u,
        p_u_raw,
        critic_cfg,
        input_dim,
        seed=seed + episode * 100,
    )
    critic_soft, ipm_soft = train_and_evaluate_critic(
        x_l,
        p_l_soft,
        x_u,
        p_u_soft,
        critic_cfg,
        input_dim,
        seed=seed + episode * 100 + 1,
    )

    row = {
        "episode": episode,
        "n_labeled": len(x_l),
        "accuracy_center": population_accuracy(p_eval_center, y_eval_center, batch),
        "accuracy_outer": population_accuracy(p_eval_outer, y_eval_outer, batch),
        "risk_center": risk_center,
        "risk_outer": risk_outer,
        "true_risk_gap": true_risk_gap,
        "mean_confidence_outer": mean_confidence(p_raw_outer, batch),
        "jacobian_raw": mean_jacobian_norm(p_raw_outer, batch),
        "jacobian_soft": mean_jacobian_norm(p_soft_outer, batch),
        "ipm_raw": ipm_raw,
        "ipm_soft": ipm_soft,
        "bound_margin_raw": ipm_raw - true_risk_gap,
        "bound_margin_soft": ipm_soft - true_risk_gap,
        "seed": seed,
    }
    return row, critic_soft


def run_active_learning(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    verbose: bool = True,
) -> list[dict]:
    device = _resolve_device(al_cfg.device)
    data = generate_active_learning_data(spiral_cfg, al_cfg)

    x_l = data["x_initial"].to(device)
    y_l = data["y_initial"].to(device)
    x_u = data["x_pool"].to(device)
    y_u_hidden = data["y_pool_hidden"].to(device)
    x_eval_center = data["x_eval_center"].to(device)
    y_eval_center = data["y_eval_center"].to(device)
    x_eval_outer = data["x_eval_outer"].to(device)
    y_eval_outer = data["y_eval_outer"].to(device)

    history: list[dict] = []

    for episode in range(al_cfg.n_episodes + 1):
        if verbose:
            print(
                f"Episode {episode:02d} | L={len(x_l)} | U={len(x_u)}",
                flush=True,
            )

        classifier = fit_classifier(
            x_l,
            y_l,
            feature_dim=al_cfg.feature_dim,
            num_classes=al_cfg.num_classes,
            seed=al_cfg.seed + episode,
            max_steps=al_cfg.classifier_max_steps,
            lr=al_cfg.classifier_lr,
            stale_patience=al_cfg.classifier_stale_patience,
            tol=al_cfg.classifier_tol,
        )

        logits_outer = logits_batched(classifier, x_eval_outer, critic_cfg.population_batch_size)
        row, critic_soft = _episode_metrics(
            classifier,
            x_l,
            y_l,
            x_u,
            x_eval_center,
            y_eval_center,
            x_eval_outer,
            y_eval_outer,
            logits_outer,
            critic_cfg,
            al_cfg,
            episode,
            al_cfg.seed,
        )
        history.append(row)

        if verbose:
            print(
                f"  acc_center={row['accuracy_center']:.4f}, "
                f"acc_outer={row['accuracy_outer']:.4f}, "
                f"gap={row['true_risk_gap']:.4f}, "
                f"ipm_raw={row['ipm_raw']:.4f}, ipm_soft={row['ipm_soft']:.4f}, "
                f"jac_raw={row['jacobian_raw']:.6f}, jac_soft={row['jacobian_soft']:.6f}",
                flush=True,
            )

        if episode == al_cfg.n_episodes:
            break

        p_u_soft = temperature_softmax(
            logits_batched(classifier, x_u, critic_cfg.population_batch_size),
            al_cfg.t_soft,
        )
        selected_local = select_topk_by_critic(
            critic_soft,
            x_u,
            p_u_soft,
            k=al_cfg.query_size,
            batch_size=critic_cfg.population_batch_size,
        )

        x_selected = x_u[selected_local]
        y_selected = y_u_hidden[selected_local]
        x_l = torch.cat([x_l, x_selected])
        y_l = torch.cat([y_l, y_selected])

        keep = torch.ones(len(x_u), dtype=torch.bool, device=device)
        keep[selected_local] = False
        x_u = x_u[keep]
        y_u_hidden = y_u_hidden[keep]

    return history


def run_multi_seed(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    seeds: list[int],
    verbose: bool = True,
) -> list[dict]:
    all_rows: list[dict] = []
    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===", flush=True)
        seed_spiral = SpiralConfig(
            r_center_min=spiral_cfg.r_center_min,
            r_center_max=spiral_cfg.r_center_max,
            r_outer_min=spiral_cfg.r_outer_min,
            r_outer_max=spiral_cfg.r_outer_max,
            perturbation=spiral_cfg.perturbation,
            seed=seed,
        )
        seed_al = ActiveLearningConfig(
            n_pool=al_cfg.n_pool,
            n_eval_center=al_cfg.n_eval_center,
            n_eval_outer=al_cfg.n_eval_outer,
            initial_labeled=al_cfg.initial_labeled,
            query_size=al_cfg.query_size,
            n_episodes=al_cfg.n_episodes,
            feature_dim=al_cfg.feature_dim,
            num_classes=al_cfg.num_classes,
            classifier_max_steps=al_cfg.classifier_max_steps,
            classifier_lr=al_cfg.classifier_lr,
            classifier_stale_patience=al_cfg.classifier_stale_patience,
            classifier_tol=al_cfg.classifier_tol,
            t_soft=al_cfg.t_soft,
            seed=seed,
            device=al_cfg.device,
        )
        seed_critic = CriticConfig(
            hidden_dim=critic_cfg.hidden_dim,
            critic_steps=critic_cfg.critic_steps,
            critic_batch_size=critic_cfg.critic_batch_size,
            critic_lr=critic_cfg.critic_lr,
            critic_weight_decay=critic_cfg.critic_weight_decay,
            validation_interval=critic_cfg.validation_interval,
            validation_size=critic_cfg.validation_size,
            critic_patience=critic_cfg.critic_patience,
            critic_tolerance=critic_cfg.critic_tolerance,
            population_batch_size=critic_cfg.population_batch_size,
            seed=seed,
        )
        all_rows.extend(run_active_learning(seed_spiral, seed_al, seed_critic, verbose=verbose))
    return all_rows
