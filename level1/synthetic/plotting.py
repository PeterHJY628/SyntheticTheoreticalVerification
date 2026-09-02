"""Plot active-learning trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _aggregate_by_episode(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in df.columns or df["seed"].nunique() == 1:
        return df.sort_values("episode")
    metrics = [
        "n_labeled",
        "accuracy",
        "boundary_accuracy",
        "true_risk_gap",
        "ipm",
        "ipm_spectral",
        "ipm_bilinear",
        "moment_gap_fro",
        "bound_slack",
        "bound_slack_spectral",
        "bound_slack_bilinear",
        "ld",
        "fd",
        "gc",
    ]
    metrics = [col for col in metrics if col in df.columns]
    grouped = df.groupby("episode")[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def plot_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_seed = "seed" in df.columns and df["seed"].nunique() > 1
    agg = _aggregate_by_episode(df)

    x = agg["n_labeled"] if "n_labeled" in agg.columns else agg["n_labeled_mean"]

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        ax.plot(x, agg["accuracy_mean"], marker="o", label="Accuracy")
        ax.fill_between(
            x,
            agg["accuracy_mean"] - agg["accuracy_std"].fillna(0),
            agg["accuracy_mean"] + agg["accuracy_std"].fillna(0),
            alpha=0.2,
        )
    else:
        ax.plot(x, agg["accuracy"], marker="o", label="Accuracy")
    ax.set_xlabel("$n_L$")
    ax.set_ylabel("Accuracy")
    ax.set_title("AL performance on target test set")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        ax.plot(x, agg["true_risk_gap_mean"], marker="o", label=r"$\Delta R_t$")
        ax.plot(x, agg["ipm_mean"], marker="s", label=r"$\widehat{\mathrm{IPM}}_t$")
        ax.fill_between(
            x,
            agg["true_risk_gap_mean"] - agg["true_risk_gap_std"].fillna(0),
            agg["true_risk_gap_mean"] + agg["true_risk_gap_std"].fillna(0),
            alpha=0.15,
        )
        ax.fill_between(
            x,
            agg["ipm_mean"] - agg["ipm_std"].fillna(0),
            agg["ipm_mean"] + agg["ipm_std"].fillna(0),
            alpha=0.15,
        )
    else:
        ax.plot(x, agg["true_risk_gap"], marker="o", label=r"$\Delta R_t$")
        ax.plot(x, agg["ipm"], marker="s", label=r"$\widehat{\mathrm{IPM}}_t$")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$n_L$")
    ax.set_ylabel("Value")
    ax.set_title(r"Theorem validation: $\Delta R_t$ vs $\widehat{\mathrm{IPM}}_t$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_risk_gap_vs_ipm.png", dpi=150)
    plt.close(fig)

    if "ipm_spectral" in df.columns and "ipm_bilinear" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, label in (
            (axes[0], "ipm_spectral", "Spectral IPM"),
            (axes[1], "ipm_bilinear", "Bilinear IPM"),
        ):
            slack_col = f"bound_slack_{col.split('_', 1)[1]}"
            if multi_seed:
                ax.plot(x, agg["true_risk_gap_mean"], marker="o", label=r"$\Delta R_t$")
                ax.plot(x, agg[f"{col}_mean"], marker="s", label=label)
                if f"{slack_col}_mean" in agg.columns:
                    ax2 = ax.twinx()
                    ax2.plot(
                        x,
                        agg[f"{slack_col}_mean"],
                        color="tab:gray",
                        linestyle="--",
                        alpha=0.7,
                        label="bound slack",
                    )
            else:
                ax.plot(x, agg["true_risk_gap"], marker="o", label=r"$\Delta R_t$")
                ax.plot(x, agg[col], marker="s", label=label)
                if slack_col in agg.columns:
                    ax2 = ax.twinx()
                    ax2.plot(
                        x,
                        agg[slack_col],
                        color="tab:gray",
                        linestyle="--",
                        alpha=0.7,
                        label="bound slack",
                    )
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("$n_L$")
            ax.set_ylabel("Value")
            ax.set_title(f"{label} vs $\\Delta R_t$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        fig.suptitle("Critic comparison: spectral vs bilinear IPM", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "figure2b_critic_comparison.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        if multi_seed:
            ax.plot(x, agg["ipm_spectral_mean"], marker="o", label="Spectral")
            ax.plot(x, agg["ipm_bilinear_mean"], marker="s", label="Bilinear")
            ax.fill_between(
                x,
                agg["ipm_spectral_mean"] - agg["ipm_spectral_std"].fillna(0),
                agg["ipm_spectral_mean"] + agg["ipm_spectral_std"].fillna(0),
                alpha=0.15,
            )
            ax.fill_between(
                x,
                agg["ipm_bilinear_mean"] - agg["ipm_bilinear_std"].fillna(0),
                agg["ipm_bilinear_mean"] + agg["ipm_bilinear_std"].fillna(0),
                alpha=0.15,
            )
        else:
            ax.plot(x, agg["ipm_spectral"], marker="o", label="Spectral")
            ax.plot(x, agg["ipm_bilinear"], marker="s", label="Bilinear")
        ax.set_xlabel("$n_L$")
        ax.set_ylabel(r"$\widehat{\mathrm{IPM}}$")
        ax.set_title("Spectral vs bilinear IPM over AL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "figure8_ipm_spectral_vs_bilinear.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if multi_seed:
        ax.plot(
            agg["moment_gap_fro_mean"],
            agg["ipm_mean"],
            marker="o",
        )
        for episode in agg["episode"]:
            row = agg[agg["episode"] == episode].iloc[0]
            ax.annotate(
                f"ep{int(episode)}",
                (row["moment_gap_fro_mean"], row["ipm_mean"]),
                fontsize=7,
                alpha=0.7,
            )
    else:
        ax.plot(agg["moment_gap_fro"], agg["ipm"], marker="o")
        for _, row in agg.iterrows():
            ax.annotate(
                f"ep{int(row['episode'])}",
                (row["moment_gap_fro"], row["ipm"]),
                fontsize=7,
                alpha=0.7,
            )
    ax.set_xlabel(r"$\|\widehat M_U - \widehat M_L\|_F$")
    ax.set_ylabel(r"$\widehat{\mathrm{IPM}}$")
    ax.set_title("Uncentered covariance mismatch vs IPM")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_moment_vs_ipm.png", dpi=150)
    plt.close(fig)

    if "ld" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        proxy_cols = ["ipm", "ld", "fd", "gc"]
        if multi_seed:
            for col, marker in zip(proxy_cols, "os^D"):
                mean_col = f"{col}_mean"
                std_col = f"{col}_std"
                if mean_col in agg.columns:
                    ax.plot(x, agg[mean_col], marker=marker, label=col.upper())
                    ax.fill_between(
                        x,
                        agg[mean_col] - agg[std_col].fillna(0),
                        agg[mean_col] + agg[std_col].fillna(0),
                        alpha=0.12,
                    )
        else:
            for col, marker in zip(proxy_cols, "os^D"):
                ax.plot(x, agg[col], marker=marker, label=col.upper())
        if multi_seed:
            ax.plot(x, agg["true_risk_gap_mean"], marker="x", label=r"$\Delta R_t$", alpha=0.7)
        else:
            ax.plot(x, agg["true_risk_gap"], marker="x", label=r"$\Delta R_t$", alpha=0.7)
        ax.set_xlabel("$n_L$")
        ax.set_ylabel("Value")
        ax.set_title("IPM vs LD / FD / GC vs true risk gap")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "figure4_proxy_comparison.png", dpi=150)
        plt.close(fig)

    if multi_seed:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, agg["bound_slack_mean"], marker="o", label="IPM - ΔR")
        ax.fill_between(
            x,
            agg["bound_slack_mean"] - agg["bound_slack_std"].fillna(0),
            agg["bound_slack_mean"] + agg["bound_slack_std"].fillna(0),
            alpha=0.2,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("$n_L$")
        ax.set_ylabel("Bound slack")
        ax.set_title(r"$\widehat{\mathrm{IPM}}_t - \Delta R_t$ (should be $\geq 0$)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "figure5_bound_slack.png", dpi=150)
        plt.close(fig)


def save_critic_traces(traces: list[dict], output_dir: Path) -> Path:
    """Persist IPM critic training curves to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "critic_traces.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(traces, handle, indent=2)
    return path


def _trace_title(trace: dict) -> str:
    if "episode" in trace:
        parts = [f"ep{int(trace['episode'])}"]
    elif "rotation_deg" in trace:
        parts = [
            f"rot{int(trace['rotation_deg'])}",
            f"s=({trace['scale_x']:.2g},{trace['scale_y']:.2g})",
        ]
    else:
        parts = ["critic"]
    if "seed" in trace:
        parts.append(f"s{int(trace['seed'])}")
    if "n_labeled" in trace:
        parts.append(f"L={int(trace['n_labeled'])}")
    if trace.get("critic_type"):
        parts.append(str(trace["critic_type"]))
    status = "conv" if trace.get("converged") else "not conv"
    parts.append(status)
    return " | ".join(parts)


def _plot_single_critic_trace(ax: plt.Axes, trace: dict) -> None:
    train_steps = trace.get("train_steps") or list(range(len(trace.get("train_loss", []))))
    train_loss = trace.get("train_loss", [])
    if train_loss:
        ax.plot(train_steps, train_loss, color="tab:blue", alpha=0.35, linewidth=0.8, label="train loss")

    val_steps = trace.get("validation_steps", [])
    val_objective = trace.get("validation_objective", [])
    if val_objective:
        val_loss = [-value for value in val_objective]
        ax.plot(
            val_steps,
            val_loss,
            color="tab:orange",
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label="val loss",
        )
        best_idx = int(max(range(len(val_objective)), key=lambda i: val_objective[i]))
        ax.scatter(
            [val_steps[best_idx]],
            [val_loss[best_idx]],
            color="tab:red",
            s=24,
            zorder=3,
            label="best val",
        )

    ax.set_title(_trace_title(trace), fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_critic_training(traces: list[dict], output_dir: Path) -> None:
    """Visualize IPM critic loss trajectories and convergence summary."""
    if not traces:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    save_critic_traces(traces, output_dir)

    n_traces = len(traces)
    ncols = min(5, n_traces)
    nrows = math.ceil(n_traces / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    for index, trace in enumerate(traces):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        _plot_single_critic_trace(ax, trace)
        if index == 0:
            ax.legend(fontsize=7, loc="upper right")
    for index in range(n_traces, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")
    fig.suptitle("IPM critic training loss trajectories", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "figure6_critic_loss_grid.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "episode": trace.get("episode"),
                "seed": trace.get("seed"),
                "rotation_deg": trace.get("rotation_deg"),
                "scale_x": trace.get("scale_x"),
                "scale_y": trace.get("scale_y"),
                "n_labeled": trace.get("n_labeled"),
                "critic_type": trace.get("critic_type"),
                "critic_steps": trace.get("critic_steps"),
                "best_validation_objective": trace.get("best_validation_objective"),
                "converged": trace.get("converged"),
                "stopped_early": trace.get("stopped_early"),
                "validation_checks": trace.get("validation_checks"),
                "train_points_logged": trace.get("train_points_logged"),
            }
            for trace in traces
        ]
    )
    summary.to_csv(output_dir / "critic_convergence_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if "episode" in summary.columns and summary["episode"].notna().any():
        x = summary["episode"]
        x_label = "Episode"
    else:
        x = summary.index
        x_label = "Run index"
    colors = ["tab:green" if c else "tab:red" for c in summary["converged"]]
    axes[0].bar(x, summary["critic_steps"], color=colors, alpha=0.85)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Training steps")
    axes[0].set_title("Critic training length (green=converged)")
    axes[0].grid(True, axis="y", alpha=0.3)

    if summary["best_validation_objective"].notna().any():
        axes[1].plot(
            x,
            summary["best_validation_objective"],
            marker="o",
            color="tab:purple",
            label="best val objective",
        )
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Best validation objective")
        axes[1].set_title("Best achieved IPM objective")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    converged_n = int(summary["converged"].sum())
    fig.suptitle(
        f"IPM critic convergence: {converged_n}/{len(summary)} episodes converged",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure7_critic_convergence.png", dpi=150)
    plt.close(fig)


def plot_single_critic_trace(trace: dict, output_dir: Path, stem: str = "critic_training") -> None:
    """Plot one critic training curve (for static baseline experiments)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_critic_traces([trace], output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_single_critic_trace(ax, trace)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (= -objective)")
    ax.legend()
    status = "converged" if trace.get("converged") else "not converged"
    ax.set_title(
        f"IPM critic training ({status}, steps={trace.get('critic_steps')}, "
        f"best obj={trace.get('best_validation_objective'):.6f})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
