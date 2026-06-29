from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#1f2933",
    "muted": "#667085",
    "blue": "#2f6f9f",
    "teal": "#2a9d8f",
    "gold": "#e9b44c",
    "rose": "#c85554",
    "violet": "#6d597a",
    "gray": "#d0d5dd",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def concept_figure(out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(0, color=PALETTE["gray"], lw=2.8, zorder=0)
    ax.arrow(0, 0, 1.85, 0, head_width=0.045, head_length=0.08, color=PALETTE["blue"], lw=2)
    ax.arrow(0, 0, 1.15, 0.95, head_width=0.045, head_length=0.08, color=PALETTE["rose"], lw=2)
    ax.plot([1.15, 1.15], [0.0, 0.95], "--", color=PALETTE["rose"], lw=1.2)
    ax.scatter([0], [0], s=48, color=PALETTE["ink"], zorder=5)
    ax.text(0.04, -0.17, "current representation", color=PALETTE["ink"])
    ax.text(1.18, 0.97, "plausible future\nnot locally executable", color=PALETTE["rose"])
    ax.text(1.15, -0.18, "action-reachable tangent set", color=PALETTE["blue"])
    ax.text(1.2, 0.43, "actionability\nresidual", color=PALETTE["rose"])
    ax.set_xlim(-0.2, 2.25)
    ax.set_ylim(-0.35, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A visual future can leave the local actionable set")
    save_figure(fig, out_base)


def geometry_projection(out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.linspace(-1.3, 1.7, 100)
    ax.plot(x, 0.45 * x, color=PALETTE["gray"], lw=3, label="Im(J)")
    delta = np.array([1.15, 1.05])
    tangent = np.array([1.25, 0.45 * 1.25])
    ax.arrow(0, 0, delta[0], delta[1], color=PALETTE["rose"], lw=2, head_width=0.04, length_includes_head=True)
    ax.arrow(0, 0, tangent[0], tangent[1], color=PALETTE["teal"], lw=2, head_width=0.04, length_includes_head=True)
    ax.plot([delta[0], tangent[0]], [delta[1], tangent[1]], "--", color=PALETTE["rose"], lw=1.3)
    ax.text(delta[0] + 0.03, delta[1] + 0.03, "imagined delta", color=PALETTE["rose"])
    ax.text(tangent[0] + 0.03, tangent[1] - 0.08, "best action delta", color=PALETTE["teal"])
    ax.text(0.98, 0.88, "residual", color=PALETTE["rose"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("representation coordinate 1")
    ax.set_ylabel("representation coordinate 2")
    ax.set_title("Projection onto the action-reachable tangent space")
    save_figure(fig, out_base)


def environment_grid(examples: dict, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.2))
    for ax, (name, data) in zip(axes, examples.items()):
        traj = data["candidate"]
        rollout = data["rollout"]
        ax.plot(traj[:, 0], traj[:, 1], color=PALETTE["rose"], lw=2, label="imagined")
        ax.plot(rollout[:, 0], rollout[:, 1], color=PALETTE["blue"], lw=2, label="executed")
        ax.scatter(traj[0, 0], traj[0, 1], s=25, color=PALETTE["ink"])
        ax.scatter(traj[-1, 0], traj[-1, 1], s=30, color=PALETTE["gold"])
        ax.set_title(name.replace("_", " "))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="upper left", frameon=False)
    save_figure(fig, out_base)


def failure_prediction_plot(scores: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    colors = {0: PALETTE["teal"], 1: PALETTE["rose"]}
    for fail, group in scores.groupby("failure"):
        axes[0].scatter(
            group["actionability_residual"],
            group["rollout_error"],
            s=12,
            alpha=0.62,
            color=colors[int(fail)],
            label="failure" if fail else "success",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("actionability residual")
    axes[0].set_ylabel("execution error")
    axes[0].legend(frameon=False)
    for env, group in scores.groupby("env"):
        order = np.argsort(group["actionability_residual"].to_numpy())
        x = np.linspace(0, 1, len(group))
        y = group.iloc[order]["failure"].rolling(max(4, len(group) // 12), min_periods=1).mean()
        axes[1].plot(x, y, lw=2, label=env.replace("_", " "))
    axes[1].set_xlabel("residual quantile")
    axes[1].set_ylabel("empirical failure rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(frameon=False, ncol=1)
    save_figure(fig, out_base)


def repair_plot(traj_npz: Path, out_base: Path) -> None:
    set_style()
    data = np.load(traj_npz, allow_pickle=True)
    names = list(data["names"])
    fig, axes = plt.subplots(1, len(names), figsize=(8.0, 2.4))
    for ax, name in zip(axes, names):
        before = data[f"{name}_before"]
        after = data[f"{name}_after"]
        rollout_before = data[f"{name}_rollout_before"]
        rollout_after = data[f"{name}_rollout_after"]
        ax.plot(before[:, 0], before[:, 1], color=PALETTE["rose"], lw=1.8, label="before")
        ax.plot(after[:, 0], after[:, 1], color=PALETTE["blue"], lw=1.8, label="repaired")
        ax.plot(rollout_before[:, 0], rollout_before[:, 1], "--", color=PALETTE["rose"], lw=1.1)
        ax.plot(rollout_after[:, 0], rollout_after[:, 1], "--", color=PALETTE["blue"], lw=1.1)
        ax.scatter(before[0, 0], before[0, 1], s=20, color=PALETTE["ink"])
        ax.scatter(before[-1, 0], before[-1, 1], s=25, color=PALETTE["gold"])
        ax.set_title(name.replace("_", " "))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    save_figure(fig, out_base)


def embodiment_heatmap(df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    envs = list(df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(8.2, 2.4), sharex=True, sharey=True)
    vmax = np.percentile(df["residual"], 92)
    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        xs = np.sort(sub["dx"].unique())
        ys = np.sort(sub["dy"].unique())
        grid = sub.pivot(index="dy", columns="dx", values="residual").loc[ys, xs].to_numpy()
        im = ax.imshow(grid, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap="magma", vmax=vmax)
        ax.set_title(env.replace("_", " "))
        ax.set_xlabel("dx")
        ax.set_aspect("equal")
    axes[0].set_ylabel("dy")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="residual")
    save_figure(fig, out_base)


def learned_j_curve(df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for env, group in df.groupby("env"):
        group = group.sort_values("samples")
        ax.plot(group["samples"], group["j_error"], marker="o", lw=2, label=env.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training transitions")
    ax.set_ylabel("relative Jacobian error")
    ax.set_title("Small local action-map models improve with data")
    ax.legend(frameon=False)
    save_figure(fig, out_base)
