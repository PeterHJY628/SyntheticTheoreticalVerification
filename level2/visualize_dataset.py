#!/usr/bin/env python3
"""Visualize the full Level 2 interleaved spiral dataset and support split."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.config import ActiveLearningConfig, SpiralConfig
from synthetic.population import generate_active_learning_data, sample_spiral_region

CLASS0_COLOR = "#4C72B0"
CLASS1_COLOR = "#C44E52"
CLASS_CMAP = ListedColormap([CLASS0_COLOR, CLASS1_COLOR])
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5], CLASS_CMAP.N)


def _spiral_xy(class_id: int, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal spiral coordinates: theta = 4πr + π·class_id."""
    theta = 4.0 * math.pi * r + math.pi * class_id
    return r * np.cos(theta), r * np.sin(theta)


def _spiral_segment(class_id: int, r_min: float, r_max: float, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    r = np.linspace(r_min, r_max, n)
    return _spiral_xy(class_id, r)


def _draw_class_spiral_arms(
    ax: plt.Axes,
    spiral_cfg: SpiralConfig,
    *,
    emphasize: str = "full",
) -> None:
    """
    Draw both class spiral curves in class colors.

    emphasize:
      - "full": both arms r∈[0.02,1] equally
      - "center": bold center segments (r≤0.25)
      - "outer": bold outer segments (r≥0.60) — highlights outer class-1/red arm ring
    """
    r_lo = 0.02
    r_hi = spiral_cfg.r_outer_max
    r_center_hi = spiral_cfg.r_center_max
    r_outer_lo = spiral_cfg.r_outer_min

    for class_id, color in ((0, CLASS0_COLOR), (1, CLASS1_COLOR)):
        # Faint full arm for context
        xf, yf = _spiral_segment(class_id, r_lo, r_hi)
        ax.plot(xf, yf, color=color, lw=1.0, alpha=0.25, zorder=1, solid_capstyle="round")

        if emphasize == "center":
            xc, yc = _spiral_segment(class_id, r_lo, r_center_hi)
            ax.plot(xc, yc, color=color, lw=2.4, alpha=0.95, zorder=2, label=f"Class {class_id} spiral (center)")
        elif emphasize == "outer":
            xo, yo = _spiral_segment(class_id, r_outer_lo, r_hi)
            ax.plot(xo, yo, color=color, lw=2.8, alpha=0.95, zorder=2, label=f"Class {class_id} spiral (outer)")
        else:
            ax.plot(xf, yf, color=color, lw=2.0, alpha=0.85, zorder=2, label=f"Class {class_id} spiral")


def _scatter_by_class(
    ax: plt.Axes,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    marker: str = "o",
    size: float = 6,
    alpha: float = 0.75,
    edgecolors: str | None = None,
    linewidths: float = 0.4,
    zorder: int = 4,
) -> None:
    kwargs: dict = dict(
        c=y.cpu().numpy(),
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        s=size,
        alpha=alpha,
        marker=marker,
        zorder=zorder,
    )
    if edgecolors is not None and marker in ("o", "s", "D", "."):
        kwargs["edgecolors"] = edgecolors
        kwargs["linewidths"] = linewidths
    ax.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), **kwargs)


def _style_spatial_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)


def _class_counts(y: torch.Tensor) -> str:
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return f"class 0: {n0}, class 1: {n1}"


def _balanced_display_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified subsample: balanced=True stores class 0 then class 1 contiguously."""
    n_per_class = n // 2
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx0 = torch.where(y == 0)[0]
    idx1 = torch.where(y == 1)[0]
    pick0 = idx0[torch.randperm(len(idx0), generator=gen)[:n_per_class]]
    pick1 = idx1[torch.randperm(len(idx1), generator=gen)[:n_per_class]]
    idx = torch.cat([pick0, pick1])
    idx = idx[torch.randperm(len(idx), generator=gen)]
    return x[idx], y[idx]


def _spiral_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], color=CLASS0_COLOR, lw=2.5, label="Class 0 spiral arm"),
        plt.Line2D([0], [0], color=CLASS1_COLOR, lw=2.5, label="Class 1 spiral arm"),
    ]


def plot_full_dataset(
    spiral_cfg: SpiralConfig,
    al_cfg: ActiveLearningConfig,
    output_dir: Path,
    n_display: int = 8000,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spiral_cfg = SpiralConfig(
        r_center_min=spiral_cfg.r_center_min,
        r_center_max=spiral_cfg.r_center_max,
        r_outer_min=spiral_cfg.r_outer_min,
        r_outer_max=spiral_cfg.r_outer_max,
        perturbation=spiral_cfg.perturbation,
        seed=seed,
    )
    al_cfg = ActiveLearningConfig(
        n_pool=al_cfg.n_pool,
        n_eval_center=al_cfg.n_eval_center,
        n_eval_outer=al_cfg.n_eval_outer,
        initial_labeled=al_cfg.initial_labeled,
        seed=seed,
    )

    data = generate_active_learning_data(spiral_cfg, al_cfg)

    pool_x, pool_y = _balanced_display_sample(
        data["x_pool"], data["y_pool_hidden"], n_display, seed=seed + 10
    )
    eval_center_x, eval_center_y = _balanced_display_sample(
        data["x_eval_center"], data["y_eval_center"], n_display, seed=seed + 20
    )
    eval_outer_x, eval_outer_y = _balanced_display_sample(
        data["x_eval_outer"], data["y_eval_outer"], n_display, seed=seed + 30
    )
    x_full, y_full, r_full = sample_spiral_region(
        n_display * 2, 0.02, 1.0, spiral_cfg.perturbation, seed=seed + 50, balanced=True
    )

    # --- Figure 1: Overview ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    _draw_class_spiral_arms(ax, spiral_cfg, emphasize="full")
    xc, yc = eval_center_x, eval_center_y
    xo, yo = eval_outer_x, eval_outer_y
    xi, yi = data["x_initial"], data["y_initial"]
    _scatter_by_class(ax, xc, yc, marker="o", size=8, alpha=0.45)
    _scatter_by_class(ax, xo, yo, marker="x", size=10, alpha=0.45)
    _scatter_by_class(ax, xi, yi, marker="o", size=140, alpha=1.0, edgecolors="black", linewidths=1.2, zorder=5)
    _style_spatial_ax(ax, "AL setup — blue/red curves = class 0/1 spiral arms")
    ax.legend(handles=_spiral_legend_handles(), loc="upper right", fontsize=9)

    ax = axes[1]
    _draw_class_spiral_arms(ax, spiral_cfg, emphasize="full")
    mask_c = r_full <= spiral_cfg.r_center_max
    mask_o = r_full >= spiral_cfg.r_outer_min
    mask_gap = ~mask_c & ~mask_o
    _scatter_by_class(ax, x_full[mask_c], y_full[mask_c], marker="o", size=5, alpha=0.5)
    ax.scatter(x_full[mask_gap, 0], x_full[mask_gap, 1], c="#bdc3c7", s=3, alpha=0.3, marker=".", zorder=3)
    _scatter_by_class(ax, x_full[mask_o], y_full[mask_o], marker="x", size=6, alpha=0.5)
    _style_spatial_ax(ax, "Full dataset — gray band = empty radial gap")
    ax.legend(handles=_spiral_legend_handles(), loc="upper right", fontsize=9)

    fig.suptitle(
        "Level 2 interleaved spirals: point color = class label; curves = ideal spiral arms",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Four panels — draw class spiral arms (outer panels bold outer segment) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    panels: list[tuple[str, torch.Tensor, torch.Tensor, str, float, str]] = [
        (
            f"Initial labeled L₀ (|L|={len(data['x_initial'])})\n{_class_counts(data['y_initial'])}",
            data["x_initial"],
            data["y_initial"],
            "o",
            100,
            "center",
        ),
        (
            f"Unlabeled pool U (sample n={len(pool_x)})\n{_class_counts(pool_y)} — both classes unlabeled",
            pool_x,
            pool_y,
            "x",
            8,
            "outer",
        ),
        (
            f"Eval center / source S\n{_class_counts(eval_center_y)}",
            eval_center_x,
            eval_center_y,
            "o",
            8,
            "center",
        ),
        (
            f"Eval outer / target X\n{_class_counts(eval_outer_y)}",
            eval_outer_x,
            eval_outer_y,
            "x",
            8,
            "outer",
        ),
    ]

    for ax, (title, x, y, marker, size, emphasize) in zip(axes.flat, panels):
        _draw_class_spiral_arms(ax, spiral_cfg, emphasize=emphasize)
        edge = "black" if size >= 50 else None
        _scatter_by_class(ax, x, y, marker=marker, size=size, alpha=0.85, edgecolors=edge, linewidths=0.8)
        _style_spatial_ax(ax, title)

    fig.suptitle(
        "Dataset splits — bold curves show active radial segment; class 1 outer arm in red",
        fontsize=13,
        y=1.01,
    )
    fig.legend(
        handles=_spiral_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=10,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Radial histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r_center = torch.linalg.vector_norm(eval_center_x, dim=1).numpy()
    r_outer = torch.linalg.vector_norm(eval_outer_x, dim=1).numpy()
    r_pool = torch.linalg.vector_norm(pool_x, dim=1).numpy()

    gap_lo, gap_hi = spiral_cfg.r_center_max, spiral_cfg.r_outer_min
    for ax in axes:
        ax.axvspan(gap_lo, gap_hi, color="#ecf0f1", alpha=0.9, label="Empty gap")
        ax.axvline(gap_lo, color="#7f8c8d", ls="--", lw=1)
        ax.axvline(gap_hi, color="#7f8c8d", ls="--", lw=1)

    axes[0].hist(r_center, bins=50, alpha=0.75, label="Center eval (S)", color=CLASS0_COLOR, density=True)
    axes[0].hist(r_outer, bins=50, alpha=0.75, label="Outer eval (X)", color=CLASS1_COLOR, density=True)
    axes[0].set_xlabel("Radial distance r")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Eval populations by radius")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 1.05)

    axes[1].hist(r_pool, bins=50, alpha=0.85, color=CLASS1_COLOR, density=True, label="Unlabeled pool U")
    axes[1].set_xlabel("Radial distance r")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Unlabeled pool r ∈ [{spiral_cfg.r_outer_min}, {spiral_cfg.r_outer_max}]")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "dataset_radial_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualizations to {output_dir}/")
    print("  - dataset_overview.png")
    print("  - dataset_splits.png")
    print("  - dataset_radial_hist.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-display", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    plot_full_dataset(
        SpiralConfig(),
        ActiveLearningConfig(),
        args.output_dir,
        n_display=args.n_display,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
