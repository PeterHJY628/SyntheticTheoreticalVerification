#!/usr/bin/env python3
"""Run synthetic active-learning experiment (buildv2)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.active_learning import run_active_learning, run_multi_seed
from synthetic.config import ActiveLearningConfig, CriticConfig, PopulationConfig
from synthetic.plotting import plot_critic_training, plot_trajectories


def save_results(rows: list[dict], output_dir: Path, critic_traces: list[dict] | None = None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    df.to_csv(output_dir / "active_learning.csv", index=False)
    with (output_dir / "active_learning.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    if "seed" in df.columns and df["seed"].nunique() > 1:
        summary = (
            df.groupby("episode")
            .agg(
                n_labeled=("n_labeled", "first"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                true_risk_gap_mean=("true_risk_gap", "mean"),
                true_risk_gap_std=("true_risk_gap", "std"),
                ipm_mean=("ipm", "mean"),
                ipm_std=("ipm", "std"),
                moment_gap_fro_mean=("moment_gap_fro", "mean"),
                moment_gap_fro_std=("moment_gap_fro", "std"),
                bound_slack_mean=("bound_slack", "mean"),
                bound_slack_std=("bound_slack", "std"),
                ld_mean=("ld", "mean"),
                ld_std=("ld", "std"),
                fd_mean=("fd", "mean"),
                fd_std=("fd", "std"),
                gc_mean=("gc", "mean"),
                gc_std=("gc", "std"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "active_learning_summary.csv", index=False)

    md_path = output_dir / "RESULTS.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Active Learning Synthetic Experiment (dino_core aligned)\n\n")
        handle.write("- Classifier: raw logits + sum-reduced Brier MSE\n")
        handle.write("- IPM input: temperature-softmax (T=2.0), detached\n")
        if "critic_type" in df.columns:
            handle.write(f"- Critic type: {df['critic_type'].iloc[0]}\n")
        if "acquisition_critic" in df.columns:
            handle.write(f"- Acquisition critic: {df['acquisition_critic'].iloc[0]}\n")
        handle.write("\n")
        handle.write(f"- Seeds: {sorted(df['seed'].unique().tolist()) if 'seed' in df.columns else [df.iloc[0].get('seed', 'N/A')]}\n")
        handle.write(f"- Episodes: 0–{int(df['episode'].max())}\n")
        handle.write(f"- Final labeled: {int(df.groupby('seed')['n_labeled'].max().mean()) if 'seed' in df.columns else int(df['n_labeled'].max())}\n\n")
        handle.write("| episode | n_labeled | accuracy | true_risk_gap | ipm | ld | fd | gc | bound_slack | moment_gap_fro | critic_steps | critic_converged |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")

        if "seed" in df.columns and df["seed"].nunique() > 1:
            summary = df.groupby("episode").mean(numeric_only=True)
            for episode, row in summary.iterrows():
                handle.write(
                    f"| {int(episode)} | {int(row['n_labeled'])} | "
                    f"{row['accuracy']:.4f} | {row['true_risk_gap']:.6f} | "
                    f"{row['ipm']:.6f} | {row['ld']:.6f} | {row['fd']:.6f} | "
                    f"{row['gc']:.6f} | {row['bound_slack']:.6f} | "
                    f"{row['moment_gap_fro']:.6f} | "
                    f"{int(row.get('critic_steps', 0))} | "
                    f"{bool(row.get('critic_converged', False))} |\n"
                )
        else:
            for _, row in df.iterrows():
                handle.write(
                    f"| {int(row['episode'])} | {int(row['n_labeled'])} | "
                    f"{row['accuracy']:.4f} | {row['true_risk_gap']:.6f} | "
                    f"{row['ipm']:.6f} | {row['ld']:.6f} | {row['fd']:.6f} | "
                    f"{row['gc']:.6f} | {row['bound_slack']:.6f} | "
                    f"{row['moment_gap_fro']:.6f} | "
                    f"{int(row.get('critic_steps', 0))} | "
                    f"{bool(row.get('critic_converged', False))} |\n"
                )

    plot_trajectories(df, output_dir)
    if critic_traces:
        plot_critic_training(critic_traces, output_dir)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Single seed (if --seeds not set)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multiple seeds for mean±std aggregation",
    )
    parser.add_argument("--n-pool", type=int, default=1_000_000)
    parser.add_argument("--n-test", type=int, default=1_000_000)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--initial-labeled", type=int, default=10)
    parser.add_argument("--rotation-deg", type=float, default=60.0)
    parser.add_argument("--scale-x", type=float, default=4.0)
    parser.add_argument("--scale-y", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bilinear-rank", type=int, default=32)
    parser.add_argument(
        "--critic-type",
        choices=("spectral", "bilinear", "both"),
        default="spectral",
        help="Critic architecture; 'both' trains spectral and bilinear each episode",
    )
    parser.add_argument(
        "--acquisition-critic",
        choices=("spectral", "bilinear"),
        default="spectral",
        help="Primary IPM column and acquisition critic when --critic-type=both",
    )
    parser.add_argument("--critic-steps", type=int, default=1_000_000)
    parser.add_argument("--proxy-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "active_learning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    pop_cfg = PopulationConfig(
        rotation_deg=args.rotation_deg,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        seed=seeds[0],
    )
    al_cfg = ActiveLearningConfig(
        n_pool=args.n_pool,
        n_test=args.n_test,
        initial_labeled=args.initial_labeled,
        query_size=args.query_size,
        n_episodes=args.n_episodes,
        proxy_chunk_size=args.proxy_chunk_size,
        seed=seeds[0],
        device=args.device,
    )
    critic_cfg = CriticConfig(
        critic_type=args.critic_type,
        acquisition_critic=args.acquisition_critic,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        critic_steps=args.critic_steps,
        seed=seeds[0],
    )

    if len(seeds) == 1:
        rows, critic_traces = run_active_learning(pop_cfg, al_cfg, critic_cfg)
    else:
        rows, critic_traces = run_multi_seed(pop_cfg, al_cfg, critic_cfg, seeds)

    df = save_results(rows, args.output_dir, critic_traces=critic_traces)
    print(f"\nSaved results and plots to {args.output_dir}")
    print(
        df[
            [
                "episode",
                "n_labeled",
                "accuracy",
                "true_risk_gap",
                "ipm",
                "ld",
                "fd",
                "gc",
                "bound_slack",
            ]
        ]
        .groupby("episode")
        .mean()
        if "seed" in df.columns and df["seed"].nunique() > 1
        else df[
            [
                "episode",
                "n_labeled",
                "accuracy",
                "true_risk_gap",
                "ipm",
                "ld",
                "fd",
                "gc",
                "bound_slack",
            ]
        ]
    )


if __name__ == "__main__":
    main()
