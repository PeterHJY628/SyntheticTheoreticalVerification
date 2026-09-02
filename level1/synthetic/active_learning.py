"""Active-learning loop for synthetic covariate shift validation."""

from __future__ import annotations

import numpy as np
import torch

from synthetic.classifier import fit_classifier, ipm_probabilities_batched, predict_batched
from synthetic.config import ActiveLearningConfig, CriticConfig, PopulationConfig
from synthetic.critic import (
    normalize_critic_config,
    resolve_critic_types,
    select_topk_by_critic,
    train_and_evaluate_critic,
)
from synthetic.metrics import moment_gap_fro, population_true_brier_risk
from synthetic.population import generate_active_learning_data
from synthetic.proxies import distribution_proxies


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ipm_column(critic_type: str) -> str:
    return f"ipm_{critic_type}"


def _bound_slack_column(critic_type: str) -> str:
    return f"bound_slack_{critic_type}"


def run_active_learning(
    pop_cfg: PopulationConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
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

    critic_cfg = normalize_critic_config(critic_cfg)
    critic_types = resolve_critic_types(critic_cfg.critic_type)

    history: list[dict] = []
    critic_traces: list[dict] = []
    batch_size = critic_cfg.population_batch_size

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

        p_l_pred = predict_batched(classifier, x_all[labeled_idx], batch_size)
        p_u_pred = predict_batched(classifier, x_all[unlabeled_idx], batch_size)
        p_test_pred = predict_batched(classifier, x_test, batch_size)

        p_l_ipm = ipm_probabilities_batched(classifier, x_all[labeled_idx], batch_size)
        p_u_ipm = ipm_probabilities_batched(classifier, x_all[unlabeled_idx], batch_size)

        y_test_pred = p_test_pred.argmax(dim=1)
        accuracy = float((y_test_pred == y_test).float().mean().item())
        bayes_test_class = p_test_true.argmax(dim=1)
        boundary_accuracy = float((y_test_pred == bayes_test_class).float().mean().item())

        true_risk_l = population_true_brier_risk(p_l_pred, p_true_all[labeled_idx])
        true_risk_u = population_true_brier_risk(p_u_pred, p_true_all[unlabeled_idx])
        true_risk_gap = true_risk_u - true_risk_l

        critics: dict[str, torch.nn.Module] = {}
        ipm_by_type: dict[str, float] = {}

        for critic_type in critic_types:
            critic, trace, ipm_value = train_and_evaluate_critic(
                critic_type,
                x_all[labeled_idx],
                p_l_ipm,
                x_all[unlabeled_idx],
                p_u_ipm,
                critic_cfg,
                al_cfg.feature_dim,
                al_cfg.num_classes,
                episode,
                batch_size,
            )
            trace.update(
                {
                    "episode": episode,
                    "seed": al_cfg.seed,
                    "n_labeled": len(labeled_idx),
                }
            )
            critic_traces.append(trace)
            critics[critic_type] = critic
            ipm_by_type[critic_type] = ipm_value

        acquisition_critic = critics[critic_cfg.acquisition_critic]
        primary_ipm = ipm_by_type[critic_cfg.acquisition_critic]

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

        row: dict = {
            "episode": episode,
            "n_labeled": len(labeled_idx),
            "accuracy": accuracy,
            "boundary_accuracy": boundary_accuracy,
            "true_risk_L": true_risk_l,
            "true_risk_U": true_risk_u,
            "true_risk_gap": true_risk_gap,
            "abs_true_risk_gap": abs(true_risk_gap),
            "ipm": primary_ipm,
            "bound_slack": primary_ipm - true_risk_gap,
            "critic_type": critic_cfg.critic_type,
            "acquisition_critic": critic_cfg.acquisition_critic,
            "moment_gap_fro": moment_gap,
            "ld": ld,
            "fd": fd,
            "gc": gc,
            "seed": al_cfg.seed,
        }

        for critic_type in critic_types:
            ipm_value = ipm_by_type[critic_type]
            trace = next(
                t
                for t in reversed(critic_traces)
                if t.get("episode") == episode and t.get("critic_type") == critic_type
            )
            row[_ipm_column(critic_type)] = ipm_value
            row[_bound_slack_column(critic_type)] = ipm_value - true_risk_gap
            row[f"critic_steps_{critic_type}"] = trace["critic_steps"]
            row[f"critic_best_validation_objective_{critic_type}"] = trace[
                "best_validation_objective"
            ]
            row[f"critic_converged_{critic_type}"] = trace["converged"]
            row[f"critic_stopped_early_{critic_type}"] = trace["stopped_early"]

        if len(critic_types) == 1:
            only = critic_types[0]
            row["critic_steps"] = row[f"critic_steps_{only}"]
            row["critic_best_validation_objective"] = row[
                f"critic_best_validation_objective_{only}"
            ]
            row["critic_converged"] = row[f"critic_converged_{only}"]
            row["critic_stopped_early"] = row[f"critic_stopped_early_{only}"]

        history.append(row)

        if verbose:
            ipm_parts = ", ".join(
                f"{name}={ipm_by_type[name]:.6f}" for name in critic_types
            )
            print(
                f"  acc={accuracy:.4f}, gap={true_risk_gap:.6f}, "
                f"ipm[{ipm_parts}], slack={row['bound_slack']:.6f}, "
                f"moment={moment_gap:.6f}, ld={ld:.6f}, fd={fd:.6f}, gc={gc:.6f}",
                flush=True,
            )

        if episode == al_cfg.n_episodes:
            break

        selected_local = select_topk_by_critic(
            acquisition_critic,
            x_all[unlabeled_idx],
            p_u_ipm,
            k=al_cfg.query_size,
            batch_size=batch_size,
        )
        selected_global = unlabeled_idx[selected_local]

        labeled_idx = torch.cat([labeled_idx, selected_global])
        keep = torch.ones(len(unlabeled_idx), dtype=torch.bool, device=device)
        keep[selected_local] = False
        unlabeled_idx = unlabeled_idx[keep]

    return history, critic_traces


def run_multi_seed(
    pop_cfg: PopulationConfig,
    al_cfg: ActiveLearningConfig,
    critic_cfg: CriticConfig,
    seeds: list[int],
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    all_traces: list[dict] = []
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
            critic_type=critic_cfg.critic_type,
            acquisition_critic=critic_cfg.acquisition_critic,
            hidden_dim=critic_cfg.hidden_dim,
            bilinear_rank=critic_cfg.bilinear_rank,
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
        history, traces = run_active_learning(seed_pop, seed_al, seed_critic, verbose=verbose)
        all_rows.extend(history)
        all_traces.extend(traces)
    return all_rows, all_traces
