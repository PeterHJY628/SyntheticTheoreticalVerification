#!/usr/bin/env python3
"""Run population-level synthetic covariate shift theorem validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.config import CriticConfig, PopulationConfig
from synthetic.critic import (
    build_critic,
    critic_training_seed,
    estimate_population_ipm,
    normalize_critic_config,
    resolve_critic_types,
    train_critic,
)
from synthetic.metrics import brier_risk_metrics, moment_shift_metrics
from synthetic.plotting import plot_critic_training, plot_single_critic_trace
from synthetic.population import generate_populations


ROTATIONS = [0, 15, 30, 45, 60, 75]
SCALES = [
    (1.0, 1.0),
    (1.5, 1.0 / 1.5),
    (2.0, 0.5),
    (3.0, 1.0 / 3.0),
    (4.0, 0.25),
]


def run_single_experiment(
    pop_cfg: PopulationConfig,
    critic_cfg: CriticConfig,
    train_critic_flag: bool = True,
) -> dict:
    torch.manual_seed(critic_cfg.seed)

    data = generate_populations(pop_cfg)
    x_source = data["x_source"]
    x_target = data["x_target"]
    p_source = data["p_source"]
    p_target = data["p_target"]
    feature_dim = x_source.shape[1]
    num_classes = p_source.shape[1]

    moment = moment_shift_metrics(x_source, x_target)
    brier = brier_risk_metrics(p_source, p_target)

    result = {
        "rotation_deg": pop_cfg.rotation_deg,
        "scale_x": pop_cfg.scale_x,
        "scale_y": pop_cfg.scale_y,
        "l2_normalize": pop_cfg.l2_normalize,
        "n_population": pop_cfg.n_population,
        "critic_type": critic_cfg.critic_type,
        **{k: v for k, v in moment.items() if isinstance(v, float)},
        **brier,
        "critic_ipm": None,
        "critic_steps": 0,
        "best_validation_objective": None,
        "critic_converged": None,
        "critic_stopped_early": None,
    }

    if not train_critic_flag:
        return result

    n = pop_cfg.n_population
    features = torch.cat([x_source, x_target], dim=0)
    probabilities = torch.cat([p_source, p_target], dim=0)
    labeled_idx = torch.arange(0, n, dtype=torch.long)
    unlabeled_idx = torch.arange(n, 2 * n, dtype=torch.long)

    critic_types = resolve_critic_types(critic_cfg.critic_type)
    traces: list[dict] = []
    primary_type = critic_types[0] if len(critic_types) == 1 else critic_cfg.acquisition_critic

    for critic_type in critic_types:
        critic = build_critic(critic_type, feature_dim, num_classes, critic_cfg, features.device)
        torch.manual_seed(critic_training_seed(critic_cfg.seed, episode=0, critic_type=critic_type))
        trace = train_critic(
            critic,
            features,
            probabilities,
            labeled_idx,
            unlabeled_idx,
            critic_cfg,
        )
        trace["critic_type"] = critic_type
        estimated_ipm = estimate_population_ipm(
            critic,
            x_source,
            p_source,
            x_target,
            p_target,
            critic_cfg,
        )
        traces.append(trace)
        result[f"critic_ipm_{critic_type}"] = float(estimated_ipm)
        result[f"critic_steps_{critic_type}"] = trace["critic_steps"]
        result[f"best_validation_objective_{critic_type}"] = trace["best_validation_objective"]
        result[f"critic_converged_{critic_type}"] = trace["converged"]
        result[f"critic_stopped_early_{critic_type}"] = trace["stopped_early"]
        if critic_type == primary_type:
            result["critic_ipm"] = float(estimated_ipm)
            result["critic_steps"] = trace["critic_steps"]
            result["best_validation_objective"] = trace["best_validation_objective"]
            result["critic_converged"] = trace["converged"]
            result["critic_stopped_early"] = trace["stopped_early"]

    result["_critic_traces"] = traces
    return result


def run_sweep(
    pop_cfg: PopulationConfig,
    critic_cfg: CriticConfig,
    rotations: list[float],
    scales: list[tuple[float, float]],
    train_critic_flag: bool,
) -> list[dict]:
    results: list[dict] = []
    total = len(rotations) * len(scales)

    for i, rotation_deg in enumerate(rotations):
        for j, (scale_x, scale_y) in enumerate(scales):
            idx = i * len(scales) + j + 1
            cfg = replace(
                pop_cfg,
                rotation_deg=rotation_deg,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            print(
                f"[{idx}/{total}] rotation={rotation_deg:.0f}°, "
                f"scale=({scale_x:.3f}, {scale_y:.3f})",
                flush=True,
            )
            start = time.time()
            row = run_single_experiment(cfg, critic_cfg, train_critic_flag)
            elapsed = time.time() - start
            row["elapsed_sec"] = round(elapsed, 2)
            results.append(row)
            print(
                f"  ||Delta M||_F={row['moment_shift_fro']:.6f}, "
                f"risk_gap={row['risk_gap']:.6f}, "
                f"critic_ipm={row['critic_ipm']}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )
    return results


def run_baseline(pop_cfg: PopulationConfig, critic_cfg: CriticConfig) -> dict:
    print("=== Baseline single experiment ===", flush=True)
    row = run_single_experiment(pop_cfg, critic_cfg, train_critic_flag=True)
    print("\nsource mean norm:", row["source_mean_norm"])
    print("target mean norm:", row["target_mean_norm"])
    print("||Delta M||_F:", row["moment_shift_fro"])
    print("R_source:", row["source_brier_risk"])
    print("R_target:", row["target_brier_risk"])
    print("risk_gap:", row["risk_gap"])
    print("critic IPM:", row["critic_ipm"])
    return row


def save_results(results: list[dict], output_dir: Path, mode: str = "sweep") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    critic_traces = []
    serializable_results = []
    for row in results:
        payload = dict(row)
        traces = payload.pop("_critic_traces", None)
        if traces:
            for trace in traces:
                trace = dict(trace)
                trace.update(
                    {
                        "rotation_deg": row["rotation_deg"],
                        "scale_x": row["scale_x"],
                        "scale_y": row["scale_y"],
                    }
                )
                critic_traces.append(trace)
        serializable_results.append(payload)

    json_path = output_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_results, handle, indent=2)

    if not serializable_results:
        return

    fieldnames = list(serializable_results[0].keys())
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_results)

    md_path = output_dir / "RESULTS.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic Population Theorem Validation\n\n")
        handle.write(
            f"- Population size: {serializable_results[0]['n_population']:,}\n"
            f"- L2 normalize: {serializable_results[0]['l2_normalize']}\n\n"
        )
        handle.write(
            "| rotation | scale_x | scale_y | ||ΔM||_F | risk_gap | critic_ipm | "
            "ipm_spectral | ipm_bilinear | critic_steps | converged |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for row in serializable_results:
            ipm = row["critic_ipm"]
            ipm_str = f"{ipm:.6f}" if ipm is not None else "N/A"
            ipm_s = row.get("critic_ipm_spectral")
            ipm_b = row.get("critic_ipm_bilinear")
            ipm_s_str = f"{ipm_s:.6f}" if ipm_s is not None else "N/A"
            ipm_b_str = f"{ipm_b:.6f}" if ipm_b is not None else "N/A"
            converged = row.get("critic_converged")
            converged_str = str(converged) if converged is not None else "N/A"
            handle.write(
                f"| {row['rotation_deg']:.0f} | {row['scale_x']:.3f} | "
                f"{row['scale_y']:.3f} | {row['moment_shift_fro']:.6f} | "
                f"{row['risk_gap']:.6f} | {ipm_str} | {ipm_s_str} | {ipm_b_str} | "
                f"{row.get('critic_steps', 0)} | {converged_str} |\n"
            )

    if critic_traces:
        if mode == "baseline" and len(critic_traces) == 1:
            plot_single_critic_trace(critic_traces[0], output_dir)
        else:
            plot_critic_training(critic_traces, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "sweep"),
        default="sweep",
        help="baseline: single cfg experiment; sweep: rotation/scale grid",
    )
    parser.add_argument("--n-population", type=int, default=1_000_000)
    parser.add_argument("--rotation-deg", type=float, default=60.0)
    parser.add_argument("--scale-x", type=float, default=4.0)
    parser.add_argument("--scale-y", type=float, default=0.25)
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--bilinear-rank", type=int, default=32)
    parser.add_argument(
        "--critic-type",
        choices=("spectral", "bilinear", "both"),
        default="spectral",
        help="Critic architecture; 'both' trains spectral and bilinear for comparison",
    )
    parser.add_argument(
        "--acquisition-critic",
        choices=("spectral", "bilinear"),
        default="spectral",
        help="Which critic IPM to use as primary / for acquisition when critic-type=both",
    )
    parser.add_argument("--critic-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-critic",
        action="store_true",
        help="Only compute moment/Brier metrics without training critic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pop_cfg = PopulationConfig(
        n_population=args.n_population,
        rotation_deg=args.rotation_deg,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        l2_normalize=args.l2_normalize,
        seed=args.seed,
    )
    critic_cfg = normalize_critic_config(
        CriticConfig(
            critic_type=args.critic_type,
            acquisition_critic=args.acquisition_critic,
            hidden_dim=args.hidden_dim,
            bilinear_rank=args.bilinear_rank,
            critic_steps=args.critic_steps,
            seed=args.seed,
        )
    )

    if args.mode == "baseline":
        results = [run_baseline(pop_cfg, critic_cfg)]
    else:
        results = run_sweep(
            pop_cfg,
            critic_cfg,
            ROTATIONS,
            SCALES,
            train_critic_flag=not args.skip_critic,
        )

    save_results(results, args.output_dir, mode=args.mode)
    print(f"\nSaved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
