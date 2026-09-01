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
    SpectralNormLayerIPM,
    estimate_population_ipm,
    train_critic,
)
from synthetic.metrics import brier_risk_metrics, moment_shift_metrics
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

    moment = moment_shift_metrics(x_source, x_target)
    brier = brier_risk_metrics(p_source, p_target)

    result = {
        "rotation_deg": pop_cfg.rotation_deg,
        "scale_x": pop_cfg.scale_x,
        "scale_y": pop_cfg.scale_y,
        "l2_normalize": pop_cfg.l2_normalize,
        "n_population": pop_cfg.n_population,
        **{k: v for k, v in moment.items() if isinstance(v, float)},
        **brier,
        "critic_ipm": None,
        "critic_steps": 0,
        "best_validation_objective": None,
    }

    if not train_critic_flag:
        return result

    n = pop_cfg.n_population
    features = torch.cat([x_source, x_target], dim=0)
    probabilities = torch.cat([p_source, p_target], dim=0)
    labeled_idx = torch.arange(0, n, dtype=torch.long)
    unlabeled_idx = torch.arange(n, 2 * n, dtype=torch.long)

    input_dim = x_source.shape[1] + p_source.shape[1]
    critic = SpectralNormLayerIPM(input_dim, critic_cfg.hidden_dim)
    trace = train_critic(
        critic,
        features,
        probabilities,
        labeled_idx,
        unlabeled_idx,
        critic_cfg,
    )
    estimated_ipm = estimate_population_ipm(
        critic,
        x_source,
        p_source,
        x_target,
        p_target,
        critic_cfg,
    )

    result.update(
        {
            "critic_ipm": float(estimated_ipm),
            "critic_steps": trace["critic_steps"],
            "best_validation_objective": trace["best_validation_objective"],
        }
    )
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


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    if not results:
        return

    fieldnames = list(results[0].keys())
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    md_path = output_dir / "RESULTS.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Synthetic Population Theorem Validation\n\n")
        handle.write(
            f"- Population size: {results[0]['n_population']:,}\n"
            f"- L2 normalize: {results[0]['l2_normalize']}\n\n"
        )
        handle.write(
            "| rotation | scale_x | scale_y | ||ΔM||_F | risk_gap | critic_ipm |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            ipm = row["critic_ipm"]
            ipm_str = f"{ipm:.6f}" if ipm is not None else "N/A"
            handle.write(
                f"| {row['rotation_deg']:.0f} | {row['scale_x']:.3f} | "
                f"{row['scale_y']:.3f} | {row['moment_shift_fro']:.6f} | "
                f"{row['risk_gap']:.6f} | {ipm_str} |\n"
            )


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
    critic_cfg = CriticConfig(
        hidden_dim=args.hidden_dim,
        critic_steps=args.critic_steps,
        seed=args.seed,
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

    save_results(results, args.output_dir)
    print(f"\nSaved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
