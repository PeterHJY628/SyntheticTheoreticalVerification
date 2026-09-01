"""Plot active-learning trajectories."""

from __future__ import annotations

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
        "moment_gap_fro",
        "bound_slack",
        "ld",
        "fd",
        "gc",
    ]
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
