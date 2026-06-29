from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#15202b",
    "muted": "#667085",
    "blue": "#2563a6",
    "sky": "#80b8d8",
    "teal": "#1f9d8a",
    "mint": "#b9e4d4",
    "gold": "#e9b44c",
    "rose": "#c94f5c",
    "rose_light": "#f3c7cc",
    "violet": "#6d597a",
    "paper": "#fbfaf8",
    "panel": "#ffffff",
    "gray": "#d0d5dd",
    "grid": "#edf0f2",
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
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["panel"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _panel(ax, label: str, title: str) -> None:
    ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.075, 1.05, title, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=PALETTE["ink"])


def _rounded_box(ax, xy, width, height, text, fc, ec=None, fontsize=8.5):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.0,
        edgecolor=ec or fc,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["ink"])
    return box


def concept_figure(out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.95))
    fig.subplots_adjust(wspace=0.34)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = axes[0]
    _panel(ax, "A", "World model dreams futures")
    _rounded_box(ax, (0.08, 0.58), 0.34, 0.20, "current\nimage", PALETTE["sky"], ec=PALETTE["blue"])
    for i, y in enumerate([0.68, 0.50, 0.32]):
        _rounded_box(ax, (0.58, y - 0.08), 0.34, 0.16, f"future {i+1}", PALETTE["panel"], ec=PALETTE["gray"], fontsize=8)
        ax.add_patch(FancyArrowPatch((0.43, 0.68), (0.58, y), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=PALETTE["muted"]))
    ax.text(0.50, 0.12, "Plausible video is not yet a control plan", ha="center", fontsize=8.5, color=PALETTE["muted"])

    ax = axes[1]
    _panel(ax, "B", "Embodiment exposes a tangent set")
    ax.scatter([0.28], [0.35], s=42, color=PALETTE["ink"], zorder=5)
    cone = Polygon([[0.28, 0.35], [0.90, 0.48], [0.88, 0.20]], closed=True, facecolor=PALETTE["mint"], edgecolor=PALETTE["teal"], alpha=0.95)
    ax.add_patch(cone)
    ax.add_patch(FancyArrowPatch((0.28, 0.35), (0.78, 0.34), arrowstyle="-|>", mutation_scale=12, lw=2.0, color=PALETTE["teal"]))
    ax.add_patch(FancyArrowPatch((0.28, 0.35), (0.63, 0.78), arrowstyle="-|>", mutation_scale=12, lw=2.0, color=PALETTE["rose"]))
    ax.plot([0.63, 0.63], [0.42, 0.78], "--", color=PALETTE["rose"], lw=1.2)
    ax.text(0.58, 0.82, "dream delta", color=PALETTE["rose"], fontsize=8.5)
    ax.text(0.58, 0.22, r"$\{J_\phi(z)u\}$", color=PALETTE["teal"], fontsize=10)
    ax.text(0.66, 0.57, "residual", color=PALETTE["rose"], fontsize=8.5)

    ax = axes[2]
    _panel(ax, "C", "Residual diagnoses and repairs")
    _rounded_box(ax, (0.07, 0.64), 0.25, 0.16, "low\nenergy", PALETTE["mint"], ec=PALETTE["teal"])
    _rounded_box(ax, (0.07, 0.27), 0.25, 0.16, "high\nenergy", PALETTE["rose_light"], ec=PALETTE["rose"])
    _rounded_box(ax, (0.58, 0.64), 0.32, 0.16, "execute", PALETTE["panel"], ec=PALETTE["teal"])
    _rounded_box(ax, (0.54, 0.27), 0.40, 0.16, "repair by\nenergy inference", PALETTE["panel"], ec=PALETTE["rose"], fontsize=8.0)
    for y, c in [(0.72, PALETTE["teal"]), (0.35, PALETTE["rose"])]:
        ax.add_patch(FancyArrowPatch((0.34, y), (0.57, y), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=c))
    ax.text(0.50, 0.11, "Jacobian maps become diagnostic interfaces", ha="center", fontsize=8.5, color=PALETTE["muted"])
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


def learned_interface_figure(learned_df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel(ax, "A", "One learned map, three roles")
    boxes = [
        ((0.04, 0.60), "exploratory\nrollouts", PALETTE["sky"]),
        ((0.36, 0.60), r"learn $J_\phi(z)$", PALETTE["mint"]),
        ((0.70, 0.74), "inverse\naction", PALETTE["panel"]),
        ((0.70, 0.50), "failure\ndiagnosis", PALETTE["panel"]),
        ((0.70, 0.26), "future\nrepair", PALETTE["panel"]),
    ]
    for xy, text, fc in boxes:
        _rounded_box(ax, xy, 0.24, 0.15, text, fc, ec=PALETTE["blue"] if fc == PALETTE["sky"] else PALETTE["teal"])
    ax.add_patch(FancyArrowPatch((0.28, 0.675), (0.36, 0.675), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=PALETTE["muted"]))
    for y in [0.815, 0.575, 0.335]:
        ax.add_patch(FancyArrowPatch((0.60, 0.675), (0.70, y), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=PALETTE["muted"]))
    ax.text(0.48, 0.14, "NJF/VERA-style action sensitivity becomes a verifier", ha="center", fontsize=8.4, color=PALETTE["muted"])

    ax = axes[1]
    _panel(ax, "B", "Few-shot local maps become usable")
    for env, group in learned_df.groupby("env"):
        group = group.sort_values("samples")
        ax.plot(group["samples"], group["j_error"], marker="o", lw=2.0, label=env.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rollout transitions")
    ax.set_ylabel("relative J error")
    ax.legend(frameon=False, fontsize=7)
    save_figure(fig, out_base)


def hostile_baseline_plot(metrics: pd.DataFrame, out_base: Path) -> None:
    set_style()
    sub = metrics[metrics["env"] == "all"].copy()
    if sub.empty:
        return
    sub["label"] = sub["score"].str.replace("_", " ", regex=False)
    sub = sub.sort_values("auroc", ascending=True)
    colors = [PALETTE["teal"] if s in ("actionability_residual", "learned_j_residual") else PALETTE["muted"] for s in sub["score"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(sub["label"], sub["auroc"], color=colors, alpha=0.95)
    ax.set_xlim(0.0, 1.03)
    ax.set_xlabel("failure-prediction AUROC")
    ax.set_title("Hostile baselines separate mechanism from pattern fitting")
    for y, val in enumerate(sub["auroc"]):
        ax.text(min(val + 0.02, 0.98), y, f"{val:.2f}", va="center", fontsize=8)
    save_figure(fig, out_base)


def robustness_grid_plot(robust: pd.DataFrame, out_base: Path) -> None:
    set_style()
    learned = robust[robust["variant"].isin(["learned_j", "noisy_learned_j"])].copy()
    envs = list(learned["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(8.4, 2.55), sharey=True)
    for ax, env in zip(axes, envs):
        sub = learned[learned["env"] == env]
        pivot = sub.pivot_table(index="noise", columns="samples", values="auroc", aggfunc="mean").sort_index()
        im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", vmin=0.45, vmax=1.0, cmap="viridis")
        ax.set_title(env.replace("_", " "), fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel("samples")
    axes[0].set_ylabel("J noise")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="AUROC")
    fig.suptitle("Robustness: learned action maps degrade gracefully, wrong maps expose failure", y=1.03, fontsize=11, fontweight="bold")
    save_figure(fig, out_base)


def visual_keypoint_plot(scores: pd.DataFrame, examples_npz: Path, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]})
    data = np.load(examples_npz, allow_pickle=True) if examples_npz.exists() else None
    for ax, name in zip(axes[:2], ["nonholonomic_car", "planar_pusher"]):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        label = "A" if name == "nonholonomic_car" else "B"
        ax.text(0.0, 1.05, f"{label}  {name.replace('_', ' ')}", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
        if data is None or name not in data["names"]:
            continue
        rep = data[f"{name}_rep"]
        pts = rep.reshape(len(rep), -1, 2)
        ax.plot(pts[:, 0, 0], pts[:, 0, 1], color=PALETTE["blue"], lw=1.8)
        ax.plot(pts[:, 1, 0], pts[:, 1, 1], color=PALETTE["rose"], lw=1.8)
        ax.scatter(pts[0, :, 0], pts[0, :, 1], s=18, color=PALETTE["ink"])
        ax.scatter(pts[-1, :, 0], pts[-1, :, 1], s=24, color=PALETTE["gold"])
    ax = axes[2]
    _panel(ax, "C", "Visual representation scores")
    metric_rows = []
    for env, group in scores.groupby("env"):
        for score in ["visual_actionability", "learned_visual_actionability", "visual_smoothness"]:
            from actionability.metrics import safe_auc

            metric_rows.append({"env": env, "score": score, "auroc": safe_auc(group["failure"], group[score])})
    metric = pd.DataFrame(metric_rows)
    labels = []
    vals = []
    colors = []
    for _, row in metric.iterrows():
        prefix = "car" if row["env"] == "nonholonomic_car" else "pusher"
        labels.append(prefix + " " + row["score"].replace("_", " ").replace("visual ", ""))
        vals.append(row["auroc"])
        colors.append(PALETTE["teal"] if "actionability" in row["score"] else PALETTE["muted"])
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("AUROC")
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
