"""Active-learning loop for synthetic covariate shift validation."""

from __future__ import annotations

import numpy as np
import torch

from synthetic.classifier import fit_classifier, predict_batched
from synthetic.config import ActiveLearningConfig, CriticConfig, PopulationConfig
from synthetic.critic import (
    SpectralNormLayerIPM,
    evaluate_ipm,
    select_topk_by_critic,
    train_ipm_critic,
)
from synthetic.metrics import moment_gap_fro, population_true_brier_risk
from synthetic.population import generate_active_learning_data
from synthetic.proxies import distribution_proxies


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_active_learning(
    pop_cfg: PopulationConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    verbose: bool = True,
) -> list[dict]:
    device = _resolve_device(al_cfg.device)
    data = generate_active_learning_data(pop_cfg, al_cfg)

    x_all = torch.cat([data["x_initial"], data["x_pool"]], dim=0).to(device)
    p_true_all = torch.cat([data["p_initial_true"], data["p_pool_true"]], dim=0).to(device)
    y_all_hidden = torch.cat([data["y_initial"], data["y_pool_hidden"]], dim=0).to(device)
    x_test = data["x_test"].to(device)
    p_test_true = data["p_test_true"].to(device)
    y_test = data["y_test"].to(device)

    labeled_idx = torch.arange(al_cfg.initial_labeled, device=device, dtype=torch.long)
    unlabeled_idx = torch.arange(
        al_cfg.initial_labeled,
        al_cfg.initial_labeled + al_cfg.n_pool,
        device=device,
        dtype=torch.long,
    )

    labels_np = y_all_hidden.cpu().numpy()
    pool_offset = al_cfg.initial_labeled
    pool_labels = labels_np[pool_offset : pool_offset + al_cfg.n_pool]
    reference_class_frequencies = (
        np.bincount(pool_labels, minlength=al_cfg.num_classes).astype(np.float64)
        / al_cfg.n_pool
    )

    input_dim = al_cfg.feature_dim + al_cfg.num_classes
    history: list[dict] = []

    for episode in range(al_cfg.n_episodes + 1):
        if verbose:
            print(
                f"Episode {episode:02d} | L={len(labeled_idx)} | U={len(unlabeled_idx)}",
                flush=True,
            )

        x_l = x_all[labeled_idx]
        y_l = y_all_hidden[labeled_idx]

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

        p_l_pred = predict_batched(classifier, x_all[labeled_idx], critic_cfg.population_batch_size)
        p_u_pred = predict_batched(classifier, x_all[unlabeled_idx], critic_cfg.population_batch_size)
        p_test_pred = predict_batched(classifier, x_test, critic_cfg.population_batch_size)

        y_test_pred = p_test_pred.argmax(dim=1)
        accuracy = float((y_test_pred == y_test).float().mean().item())
        bayes_test_class = p_test_true.argmax(dim=1)
        boundary_accuracy = float((y_test_pred == bayes_test_class).float().mean().item())

        true_risk_l = population_true_brier_risk(p_l_pred, p_true_all[labeled_idx])
        true_risk_u = population_true_brier_risk(p_u_pred, p_true_all[unlabeled_idx])
        true_risk_gap = true_risk_u - true_risk_l

        critic = SpectralNormLayerIPM(input_dim, critic_cfg.hidden_dim).to(device)
        torch.manual_seed(critic_cfg.seed + episode)
        train_ipm_critic(
            critic,
            x_all[labeled_idx],
            p_l_pred.detach(),
            x_all[unlabeled_idx],
            p_u_pred.detach(),
            critic_cfg,
        )

        ipm_value = evaluate_ipm(
            critic,
            x_all[labeled_idx],
            p_l_pred,
            x_all[unlabeled_idx],
            p_u_pred,
            critic_cfg.population_batch_size,
        )

        moment_gap = moment_gap_fro(x_all[labeled_idx], x_all[unlabeled_idx])

        labeled_np = labeled_idx.cpu().numpy()
        ld, fd, gc = distribution_proxies(
            x_all,
            labels_np,
            labeled_np,
            reference_class_frequencies,
            al_cfg.num_classes,
            al_cfg.proxy_chunk_size,
        )

        row = {
            "episode": episode,
            "n_labeled": len(labeled_idx),
            "accuracy": accuracy,
            "boundary_accuracy": boundary_accuracy,
            "true_risk_L": true_risk_l,
            "true_risk_U": true_risk_u,
            "true_risk_gap": true_risk_gap,
            "abs_true_risk_gap": abs(true_risk_gap),
            "ipm": ipm_value,
            "bound_slack": ipm_value - true_risk_gap,
            "moment_gap_fro": moment_gap,
            "ld": ld,
            "fd": fd,
            "gc": gc,
            "seed": al_cfg.seed,
        }
        history.append(row)

        if verbose:
            print(
                f"  acc={accuracy:.4f}, gap={true_risk_gap:.6f}, "
                f"ipm={ipm_value:.6f}, slack={row['bound_slack']:.6f}, "
                f"moment={moment_gap:.6f}, ld={ld:.6f}, fd={fd:.6f}, gc={gc:.6f}",
                flush=True,
            )

        if episode == al_cfg.n_episodes:
            break

        selected_local = select_topk_by_critic(
            critic,
            x_all[unlabeled_idx],
            p_u_pred,
            k=al_cfg.query_size,
            batch_size=critic_cfg.population_batch_size,
        )
        selected_global = unlabeled_idx[selected_local]

        labeled_idx = torch.cat([labeled_idx, selected_global])
        keep = torch.ones(len(unlabeled_idx), dtype=torch.bool, device=device)
        keep[selected_local] = False
        unlabeled_idx = unlabeled_idx[keep]

    return history


def run_multi_seed(
    pop_cfg: PopulationConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    seeds: list[int],
    verbose: bool = True,
) -> list[dict]:
    all_rows: list[dict] = []
    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===", flush=True)
        seed_pop = PopulationConfig(
            n_population=pop_cfg.n_population,
            rotation_deg=pop_cfg.rotation_deg,
            scale_x=pop_cfg.scale_x,
            scale_y=pop_cfg.scale_y,
            beta=pop_cfg.beta,
            l2_normalize=pop_cfg.l2_normalize,
            seed=seed,
        )
        seed_al = ActiveLearningConfig(
            n_pool=al_cfg.n_pool,
            n_test=al_cfg.n_test,
            initial_labeled=al_cfg.initial_labeled,
            initial_sample_buffer=al_cfg.initial_sample_buffer,
            query_size=al_cfg.query_size,
            n_episodes=al_cfg.n_episodes,
            proxy_chunk_size=al_cfg.proxy_chunk_size,
            feature_dim=al_cfg.feature_dim,
            num_classes=al_cfg.num_classes,
            classifier_max_steps=al_cfg.classifier_max_steps,
            classifier_lr=al_cfg.classifier_lr,
            classifier_stale_patience=al_cfg.classifier_stale_patience,
            classifier_tol=al_cfg.classifier_tol,
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
        all_rows.extend(run_active_learning(seed_pop, seed_al, seed_critic, verbose=verbose))
    return all_rows
