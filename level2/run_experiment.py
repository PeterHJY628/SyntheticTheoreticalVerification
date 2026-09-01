#!/usr/bin/env python3
"""Run Level 2 density-gap synthetic experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.ablations import run_density_gap_ablation, run_temperature_ablation
from synthetic.active_learning import run_active_learning
from synthetic.config import ActiveLearningConfig, CriticConfig, ExperimentConfig, SpiralConfig
from synthetic.plotting import plot_all_figures


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in df.columns or df["seed"].nunique() == 1:
        return df
    metrics = [
        c
        for c in df.columns
        if c not in ("episode", "seed", "n_labeled")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    summary = df.groupby("episode")[metrics + ["n_labeled"]].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    return summary.reset_index()


def save_config(cfg: ExperimentConfig, output_dir: Path) -> None:
    payload = {
        "spiral": asdict(cfg.spiral),
        "al": asdict(cfg.al),
        "critic": asdict(cfg.critic),
        "seeds": cfg.seeds,
        "density_gap_r_mins": cfg.density_gap_r_mins,
        "temperature_values": cfg.temperature_values,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-pool", type=int, default=200_000)
    parser.add_argument("--n-eval", type=int, default=200_000)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--t-soft", type=float, default=5.0)
    parser.add_argument("--critic-steps", type=int, default=40_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Smaller populations for smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_pool = 5_000 if args.quick else args.n_pool
    n_eval = 20_000 if args.quick else args.n_eval
    critic_steps = 8_000 if args.quick else args.critic_steps
    seeds = [0] if args.quick else args.seeds
    n_episodes = 2 if args.quick else args.n_episodes

    spiral_cfg = SpiralConfig()
    al_cfg = ActiveLearningConfig(
        n_pool=n_pool,
        n_eval_center=n_eval,
        n_eval_outer=n_eval,
        n_episodes=n_episodes,
        t_soft=args.t_soft,
        device=args.device,
    )
    critic_cfg = CriticConfig(critic_steps=critic_steps)
    cfg = ExperimentConfig(spiral=spiral_cfg, al=al_cfg, critic=critic_cfg, seeds=seeds)

    save_config(cfg, output_dir)

    if len(seeds) == 1:
        rows = run_active_learning(spiral_cfg, al_cfg, critic_cfg)
    else:
        rows = []
        for seed in seeds:
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
            seed_rows = run_active_learning(seed_spiral, seed_al, seed_critic)
            rows.extend(seed_rows)
            pd.DataFrame(rows).to_csv(output_dir / "level2_metrics_all_seeds.csv", index=False)
            aggregate_summary(pd.DataFrame(rows)).to_csv(
                output_dir / "level2_metrics_summary.csv", index=False
            )
            plot_all_figures(pd.DataFrame(rows), output_dir, spiral_cfg)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "level2_metrics_all_seeds.csv", index=False)
    summary = aggregate_summary(df)
    summary.to_csv(output_dir / "level2_metrics_summary.csv", index=False)

    if not args.skip_ablations:
        print("\n=== Density Gap Ablation ===", flush=True)
        density_rows: list[dict] = []
        for seed in seeds[: min(3, len(seeds))]:
            print(f"Seed {seed}", flush=True)
            density_rows.extend(
                run_density_gap_ablation(
                    spiral_cfg,
                    al_cfg,
                    critic_cfg,
                    cfg.density_gap_r_mins,
                    seed=seed,
                )
            )
        pd.DataFrame(density_rows).to_csv(
            output_dir / "level2_density_gap_ablation.csv", index=False
        )

        print("\n=== Temperature Ablation ===", flush=True)
        temp_rows: list[dict] = []
        for seed in seeds[: min(3, len(seeds))]:
            print(f"Seed {seed}", flush=True)
            temp_rows.extend(
                run_temperature_ablation(
                    spiral_cfg,
                    al_cfg,
                    critic_cfg,
                    cfg.temperature_values,
                    seed=seed,
                )
            )
        pd.DataFrame(temp_rows).to_csv(
            output_dir / "level2_temperature_ablation.csv", index=False
        )

    plot_all_figures(df, output_dir, spiral_cfg)
    print(f"\nSaved results to {output_dir}")
    print(summary if len(summary) > 0 else df.groupby("episode").mean(numeric_only=True))


if __name__ == "__main__":
    main()
