#!/usr/bin/env python3
"""Parse partial run.log into structured results (episodes 0..max_episode)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_active_learning import save_results

EP_RE = re.compile(r"Episode\s+(\d+)\s+\|\s+L=(\d+)\s+\|\s+U=(\d+)")
METRIC_RE = re.compile(
    r"acc=([\d.]+),\s+gap=([\d.eE+-]+),\s+ipm=([\d.eE+-]+),\s+slack=([\d.eE+-]+),\s+"
    r"moment=([\d.eE+-]+),\s+ld=([\d.eE+-]+),\s+fd=([\d.eE+-]+),\s+gc=([\d.eE+-]+)"
)


def parse_run_log(log_path: Path, max_episode: int, seed: int = 42) -> list[dict]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    i = 0
    while i < len(lines):
        ep_match = EP_RE.search(lines[i])
        if ep_match is None:
            i += 1
            continue
        episode = int(ep_match.group(1))
        if episode > max_episode:
            break
        n_labeled = int(ep_match.group(2))
        n_unlabeled = int(ep_match.group(3))
        if i + 1 >= len(lines):
            break
        metric_match = METRIC_RE.search(lines[i + 1])
        if metric_match is None:
            break
        gap = float(metric_match.group(2))
        rows.append(
            {
                "episode": episode,
                "n_labeled": n_labeled,
                "n_unlabeled": n_unlabeled,
                "accuracy": float(metric_match.group(1)),
                "true_risk_gap": gap,
                "abs_true_risk_gap": abs(gap),
                "ipm": float(metric_match.group(3)),
                "bound_slack": float(metric_match.group(4)),
                "moment_gap_fro": float(metric_match.group(5)),
                "ld": float(metric_match.group(6)),
                "fd": float(metric_match.group(7)),
                "gc": float(metric_match.group(8)),
                "seed": seed,
            }
        )
        i += 2
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results" / "active_learning_rerun" / "run.log",
    )
    parser.add_argument("--max-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "active_learning_rerun",
    )
    args = parser.parse_args()

    rows = parse_run_log(args.log, args.max_episode, args.seed)
    if not rows:
        raise SystemExit(f"No complete episodes found in {args.log}")

    df = save_results(rows, args.output_dir)
    print(f"Parsed {len(rows)} episodes (0–{rows[-1]['episode']})")
    print(f"Saved to {args.output_dir}")
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
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
