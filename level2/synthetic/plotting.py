"""Plot Level 2 experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from synthetic.config import SpiralConfig
from synthetic.population import sample_spiral_region


def _aggregate_by_episode(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in df.columns or df["seed"].nunique() == 1:
        return df.sort_values("episode")
    metrics = [
        "n_labeled",
        "accuracy_center",
        "accuracy_outer",
        "true_risk_gap",
        "ipm_raw",
        "ipm_soft",
        "jacobian_raw",
        "jacobian_soft",
        "bound_margin_raw",
        "bound_margin_soft",
        "mean_confidence_outer",
    ]
    grouped = df.groupby("episode")[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    return grouped.reset_index()


def _plot_with_band(ax, x, mean, std, label, marker="o"):
    ax.plot(x, mean, marker=marker, label=label)
    if std is not None:
        ax.fill_between(
            x,
            mean - std.fillna(0),
            mean + std.fillna(0),
            alpha=0.15,
        )


def plot_geometry(
    spiral_cfg: SpiralConfig,
    x_labeled: torch.Tensor | None,
    y_labeled: torch.Tensor | None,
    output_path: Path,
    n_display: int = 5000,
) -> None:
    """Scatter plot showing center vs outer support gap."""
    x_center, y_center, _ = sample_spiral_region(
        n_display,
        spiral_cfg.r_center_min,
        spiral_cfg.r_center_max,
        spiral_cfg.perturbation,
        seed=spiral_cfg.seed,
        balanced=True,
    )
    x_outer, y_outer, _ = sample_spiral_region(
        n_display,
        spiral_cfg.r_outer_min,
        spiral_cfg.r_outer_max,
        spiral_cfg.perturbation,
        seed=spiral_cfg.seed + 1,
        balanced=True,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        x_center[:, 0],
        x_center[:, 1],
        c=y_center,
        cmap="coolwarm",
        s=4,
        alpha=0.5,
        label="Center (labeled support)",
    )
    ax.scatter(
        x_outer[:, 0],
        x_outer[:, 1],
        c=y_outer,
        cmap="coolwarm",
        s=4,
        alpha=0.5,
        marker="x",
        label="Outer (unlabeled pool)",
    )
    if x_labeled is not None and y_labeled is not None:
        ax.scatter(
            x_labeled[:, 0].cpu(),
            x_labeled[:, 1].cpu(),
            c=y_labeled.cpu(),
            cmap="coolwarm",
            s=40,
            edgecolors="black",
            linewidths=0.5,
            label="Initial labeled",
        )
    ax.set_aspect("equal")
    ax.set_title("Interleaved Spirals: Density Gap")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all_figures(
    df: pd.DataFrame,
    output_dir: Path,
    spiral_cfg: SpiralConfig | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_seed = "seed" in df.columns and df["seed"].nunique() > 1
    agg = _aggregate_by_episode(df)
    x = agg["n_labeled"] if "n_labeled" in agg.columns else agg["n_labeled_mean"]

    if spiral_cfg is not None:
        plot_geometry(spiral_cfg, None, None, output_dir / "level2_geometry.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        _plot_with_band(ax, x, agg["accuracy_outer_mean"], agg["accuracy_outer_std"], "Outer accuracy")
        _plot_with_band(
            ax, x, agg["accuracy_center_mean"], agg["accuracy_center_std"], "Center accuracy", "s"
        )
    else:
        ax.plot(x, agg["accuracy_outer"], marker="o", label="Outer accuracy")
        ax.plot(x, agg["accuracy_center"], marker="s", label="Center accuracy")
    ax.set_xlabel("$n_{\\mathrm{labeled}}$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Active Learning Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "level2_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        _plot_with_band(ax, x, agg["true_risk_gap_mean"], agg["true_risk_gap_std"], r"$\Delta R_t$")
        _plot_with_band(ax, x, agg["ipm_raw_mean"], agg["ipm_raw_std"], r"IPM$^{\mathrm{raw}}$", "s")
        _plot_with_band(ax, x, agg["ipm_soft_mean"], agg["ipm_soft_std"], r"IPM$^{\mathrm{soft}}$", "^")
    else:
        ax.plot(x, agg["true_risk_gap"], marker="o", label=r"$\Delta R_t$")
        ax.plot(x, agg["ipm_raw"], marker="s", label=r"IPM$^{\mathrm{raw}}$")
        ax.plot(x, agg["ipm_soft"], marker="^", label=r"IPM$^{\mathrm{soft}}$")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$n_{\\mathrm{labeled}}$")
    ax.set_ylabel("Value")
    ax.set_title(r"True Risk Gap vs Raw/Soft IPM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "level2_risk_ipm.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        _plot_with_band(ax, x, agg["jacobian_raw_mean"], agg["jacobian_raw_std"], r"$J^{\mathrm{raw}}$")
        _plot_with_band(ax, x, agg["jacobian_soft_mean"], agg["jacobian_soft_std"], r"$J^{\mathrm{soft}}$", "s")
        ax2 = ax.twinx()
        _plot_with_band(
            ax2,
            x,
            agg["mean_confidence_outer_mean"],
            agg["mean_confidence_outer_std"],
            "Confidence",
            "^",
        )
        ax2.set_ylabel("Mean max probability")
    else:
        ax.plot(x, agg["jacobian_raw"], marker="o", label=r"$J^{\mathrm{raw}}$")
        ax.plot(x, agg["jacobian_soft"], marker="s", label=r"$J^{\mathrm{soft}}$")
        ax2 = ax.twinx()
        ax2.plot(x, agg["mean_confidence_outer"], marker="^", color="green", label="Confidence", alpha=0.7)
        ax2.set_ylabel("Mean max probability")
    ax.set_xlabel("$n_{\\mathrm{labeled}}$")
    ax.set_ylabel("Jacobian Frobenius norm")
    ax.set_title("Jacobian Collapse vs Temperature Softening")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "level2_jacobian.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        _plot_with_band(
            ax, x, agg["bound_margin_raw_mean"], agg["bound_margin_raw_std"], r"$B^{\mathrm{raw}}$"
        )
        _plot_with_band(
            ax, x, agg["bound_margin_soft_mean"], agg["bound_margin_soft_std"], r"$B^{\mathrm{soft}}$", "s"
        )
    else:
        ax.plot(x, agg["bound_margin_raw"], marker="o", label=r"$B^{\mathrm{raw}}$")
        ax.plot(x, agg["bound_margin_soft"], marker="s", label=r"$B^{\mathrm{soft}}$")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$n_{\\mathrm{labeled}}$")
    ax.set_ylabel("Bound margin (IPM - ΔR)")
    ax.set_title("Bound Margin: Raw vs Softened")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "level2_bound_margin.png", dpi=150)
    plt.close(fig)
