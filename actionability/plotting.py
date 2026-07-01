from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#15202b",
    "black": "#111827",
    "muted": "#667085",
    "soft_text": "#4b5563",
    "blue": "#2563a6",
    "blue_dark": "#0b4f8a",
    "blue_mid": "#3f94c5",
    "blue_light": "#9ecae1",
    "blue_pale": "#d8ecf6",
    "sky": "#80b8d8",
    "teal": "#1f9d8a",
    "mint": "#b9e4d4",
    "gold": "#e9b44c",
    "orange": "#f3a34d",
    "rose": "#c94f5c",
    "rose_light": "#f3c7cc",
    "violet": "#6d597a",
    "paper": "#ffffff",
    "wash": "#f6f8fb",
    "panel": "#ffffff",
    "gray": "#d0d5dd",
    "gray_dark": "#7b8798",
    "gray_mid": "#b8c0cc",
    "gray_light": "#e8edf3",
    "grid": "#edf0f2",
}

_MAGMA = mpl.colormaps["magma"] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("magma")
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "warm_residual",
    [_MAGMA(x) for x in np.linspace(0.11, 0.96, 256)],
)

METHOD_COLORS = {
    "actionability_residual": PALETTE["blue_dark"],
    "learned_j_residual": PALETTE["blue_mid"],
    "noisy_j_residual": PALETTE["blue_light"],
    "learned_classifier": "#154f92",
    "mean_residual": PALETTE["blue_mid"],
    "idm_reconstruction_error": PALETTE["gray_dark"],
    "wav_proxy": PALETTE["gray_mid"],
    "wrong_j_residual": PALETTE["orange"],
    "smoothness": PALETTE["gray_mid"],
    "passive_nll": PALETTE["gray_mid"],
    "random_score": PALETTE["gray_mid"],
}

ENV_COLORS = {
    "point_mass": PALETTE["gray_dark"],
    "nonholonomic_car": PALETTE["blue_dark"],
    "two_link_arm": PALETTE["teal"],
    "planar_pusher": PALETTE["orange"],
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
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
    ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=9.7, fontweight="bold", color=PALETTE["black"])
    ax.text(0.075, 1.05, title, transform=ax.transAxes, fontsize=9.1, fontweight="bold", color=PALETTE["black"])


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


def _clean_numeric_axis(ax, y_label: str | None = None, x_label: str | None = None) -> None:
    ax.spines["left"].set_color(PALETTE["gray_mid"])
    ax.spines["bottom"].set_color(PALETTE["gray_mid"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.tick_params(axis="both", colors=PALETTE["soft_text"], length=2.5, width=0.7)
    if y_label:
        ax.set_ylabel(y_label, color=PALETTE["soft_text"])
    if x_label:
        ax.set_xlabel(x_label, color=PALETTE["soft_text"])


def _metric_bar_panel(ax, labels, values, colors, title: str, ylim=(0.35, 1.04), ylabel: str | None = None) -> None:
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.58, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=8.8, color=PALETTE["black"], pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    _clean_numeric_axis(ax, y_label=ylabel)
    ax.axhline(0.5, ls="--", lw=0.8, color=PALETTE["gray_mid"], zorder=0)
    ax.text(len(values) - 0.35, 0.515, "chance", fontsize=6.8, color=PALETTE["gray_dark"], ha="right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.025, ylim[1] - 0.015),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=PALETTE["black"],
            fontweight="bold" if val >= 0.9 else "normal",
        )


def _set_equal_with_margin(ax, arrays, margin=0.08) -> None:
    pts = np.concatenate([a[:, :2] for a in arrays if len(a)], axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    pad = span * (0.55 + margin)
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _card_background(ax) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=PALETTE["wash"],
            edgecolor=PALETTE["gray_light"],
            linewidth=0.8,
            zorder=-10,
        )
    )


def concept_figure(out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.7, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panels = [
        (0.025, "1", "world model\ndreams"),
        (0.275, "2", "body tests\nlocal motion"),
        (0.525, "3", "residual\nflags failure"),
        (0.775, "4", "repair becomes\nexecutable"),
    ]
    w, y0, h = 0.205, 0.16, 0.70
    for x0, num, title in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=PALETTE["wash"],
                edgecolor=PALETTE["gray_light"],
                linewidth=0.9,
            )
        )
        ax.text(x0 + 0.02, y0 + h - 0.06, num, fontsize=10, fontweight="bold", color=PALETTE["blue_dark"])
        ax.text(x0 + 0.052, y0 + h - 0.065, title, fontsize=8.8, fontweight="bold", color=PALETTE["black"], va="top")
    for x0 in [0.235, 0.485, 0.735]:
        ax.add_patch(FancyArrowPatch((x0, 0.51), (x0 + 0.032, 0.51), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=PALETTE["gray_dark"]))

    # Panel 1: candidate futures.
    x0 = 0.025
    ax.scatter([x0 + 0.055], [0.39], s=35, color=PALETTE["black"], zorder=5)
    futures = [
        [(x0 + 0.07, 0.39), (x0 + 0.125, 0.57), (x0 + 0.175, 0.62)],
        [(x0 + 0.07, 0.39), (x0 + 0.14, 0.43), (x0 + 0.18, 0.36)],
        [(x0 + 0.07, 0.39), (x0 + 0.115, 0.28), (x0 + 0.17, 0.25)],
    ]
    for pts, c in zip(futures, [PALETTE["blue_light"], PALETTE["blue_mid"], PALETTE["rose"]]):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c, lw=2.0, solid_capstyle="round")
        ax.scatter([xs[-1]], [ys[-1]], s=22, color=c, edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(x0 + 0.03, 0.22, "plausible future\nis not yet a plan", fontsize=7.0, color=PALETTE["soft_text"])

    # Panel 2: tangent cone and residual.
    x0 = 0.275
    origin = (x0 + 0.065, 0.39)
    cone = Polygon(
        [origin, (x0 + 0.182, 0.50), (x0 + 0.178, 0.30)],
        closed=True,
        facecolor=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue_mid"],
        linewidth=1.2,
    )
    ax.add_patch(cone)
    ax.scatter([origin[0]], [origin[1]], s=35, color=PALETTE["black"], zorder=5)
    ax.add_patch(FancyArrowPatch(origin, (x0 + 0.153, 0.405), arrowstyle="-|>", mutation_scale=11, lw=2.0, color=PALETTE["blue_dark"]))
    dream = (x0 + 0.145, 0.62)
    ax.add_patch(FancyArrowPatch(origin, dream, arrowstyle="-|>", mutation_scale=11, lw=2.0, color=PALETTE["rose"]))
    ax.plot([dream[0], x0 + 0.151], [dream[1], 0.49], "--", color=PALETTE["rose"], lw=1.1)
    ax.text(x0 + 0.105, 0.23, r"$\{J_\phi(z)u\}$", fontsize=8.8, color=PALETTE["blue_dark"])
    ax.text(x0 + 0.127, 0.64, "dream", fontsize=6.8, color=PALETTE["rose"])

    # Panel 3: residual decision.
    x0 = 0.525
    _rounded_box(ax, (x0 + 0.035, 0.54), 0.072, 0.12, "low", PALETTE["blue_pale"], ec=PALETTE["blue_mid"], fontsize=7.5)
    _rounded_box(ax, (x0 + 0.035, 0.32), 0.072, 0.12, "high", PALETTE["rose_light"], ec=PALETTE["rose"], fontsize=7.5)
    ax.add_patch(FancyArrowPatch((x0 + 0.113, 0.60), (x0 + 0.172, 0.60), arrowstyle="-|>", mutation_scale=10, lw=1.6, color=PALETTE["blue_mid"]))
    ax.add_patch(FancyArrowPatch((x0 + 0.113, 0.38), (x0 + 0.172, 0.38), arrowstyle="-|>", mutation_scale=10, lw=1.6, color=PALETTE["rose"]))
    ax.text(x0 + 0.13, 0.64, "execute", fontsize=7.2, color=PALETTE["blue_dark"])
    ax.text(x0 + 0.13, 0.42, "repair", fontsize=7.2, color=PALETTE["rose"])
    ax.text(x0 + 0.035, 0.22, "energy is a\nverifier, not decoration", fontsize=7.0, color=PALETTE["soft_text"])

    # Panel 4: repaired future.
    x0 = 0.775
    start = (x0 + 0.045, 0.33)
    goal = (x0 + 0.172, 0.61)
    ax.plot([start[0], x0 + 0.12, goal[0]], [start[1], 0.62, goal[1]], "--", color=PALETTE["rose"], lw=1.4, alpha=0.75)
    ax.plot([start[0], x0 + 0.10, goal[0]], [start[1], 0.43, goal[1]], color=PALETTE["blue_dark"], lw=2.2)
    ax.scatter([start[0]], [start[1]], s=32, color=PALETTE["black"], zorder=4)
    ax.scatter([goal[0]], [goal[1]], s=40, color=PALETTE["orange"], edgecolor="white", linewidth=0.7, zorder=5)
    _rounded_box(ax, (x0 + 0.045, 0.20), 0.13, 0.08, "actionable", PALETTE["blue_pale"], ec=PALETTE["blue_mid"], fontsize=7.2)
    ax.text(0.5, 0.045, "Actionability turns local Jacobians into a diagnostic interface for imagined robot futures.", ha="center", fontsize=8.4, color=PALETTE["soft_text"])
    save_figure(fig, out_base)


def geometry_projection(out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.linspace(-1.3, 1.7, 100)
    ax.plot(x, 0.45 * x, color=PALETTE["gray_mid"], lw=3, label="reachable line")
    delta = np.array([1.15, 1.05])
    tangent = np.array([1.25, 0.45 * 1.25])
    ax.arrow(0, 0, delta[0], delta[1], color=PALETTE["rose"], lw=2, head_width=0.04, length_includes_head=True)
    ax.arrow(0, 0, tangent[0], tangent[1], color=PALETTE["blue_dark"], lw=2, head_width=0.04, length_includes_head=True)
    ax.plot([delta[0], tangent[0]], [delta[1], tangent[1]], "--", color=PALETTE["rose"], lw=1.3)
    ax.text(delta[0] + 0.03, delta[1] + 0.03, "imagined delta", color=PALETTE["rose"])
    ax.text(tangent[0] + 0.03, tangent[1] - 0.08, "best action delta", color=PALETTE["blue_dark"])
    ax.text(0.98, 0.88, "residual", color=PALETTE["rose"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("representation coordinate 1")
    ax.set_ylabel("representation coordinate 2")
    ax.set_title("Projection onto the action-reachable tangent space")
    _clean_numeric_axis(ax)
    save_figure(fig, out_base)


def learned_interface_figure(learned_df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.subplots_adjust(wspace=0.30)
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _card_background(ax)
    _panel(ax, "A", "One learned map, three roles")
    boxes = [
        ((0.05, 0.60), "exploratory\nrollouts", PALETTE["blue_light"]),
        ((0.37, 0.60), r"learn $J_\phi(z)$", PALETTE["blue_pale"]),
        ((0.70, 0.74), "inverse\naction", PALETTE["panel"]),
        ((0.70, 0.50), "failure\ndiagnosis", PALETTE["panel"]),
        ((0.70, 0.26), "future\nrepair", PALETTE["panel"]),
    ]
    for xy, text, fc in boxes:
        edge = PALETTE["blue_dark"] if fc == PALETTE["blue_light"] else PALETTE["blue_mid"]
        _rounded_box(ax, xy, 0.23, 0.15, text, fc, ec=edge, fontsize=8.0)
    ax.add_patch(FancyArrowPatch((0.29, 0.675), (0.37, 0.675), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=PALETTE["gray_dark"]))
    for y in [0.815, 0.575, 0.335]:
        ax.add_patch(FancyArrowPatch((0.60, 0.675), (0.70, y), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=PALETTE["gray_dark"]))
    ax.text(0.50, 0.14, "visual action sensitivity becomes a verifier", ha="center", fontsize=7.4, color=PALETTE["soft_text"])

    ax = axes[1]
    _panel(ax, "B", "Few-shot local maps become usable")
    for env, group in learned_df.groupby("env"):
        group = group.sort_values("samples")
        color = ENV_COLORS.get(env, PALETTE["blue_mid"])
        ax.plot(group["samples"], group["j_error"], marker="o", lw=2.0, ms=4.0, color=color)
        ax.text(group["samples"].iloc[-1] * 1.05, group["j_error"].iloc[-1], env.replace("_", " "), color=color, fontsize=6.8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rollout transitions")
    ax.set_ylabel("relative J error")
    ax.set_xlim(learned_df["samples"].min() * 0.9, learned_df["samples"].max() * 1.70)
    _clean_numeric_axis(ax)
    save_figure(fig, out_base)


def hostile_baseline_plot(metrics: pd.DataFrame, out_base: Path) -> None:
    set_style()
    sub = metrics[metrics["env"] == "all"].copy()
    if sub.empty:
        return
    by_score = sub.set_index("score")["auroc"].to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.subplots_adjust(wspace=0.34)

    primary = [
        ("classifier", "learned_classifier"),
        ("analytic", "actionability_residual"),
        ("learned J", "learned_j_residual"),
    ]
    _metric_bar_panel(
        axes[0],
        [a for a, _ in primary],
        [by_score[b] for _, b in primary],
        [METHOD_COLORS[b] for _, b in primary],
        "Mechanism vs pattern fit",
        ylabel="AUROC",
    )

    learned = [
        ("analytic", "actionability_residual"),
        ("learned J", "learned_j_residual"),
        ("noisy J", "noisy_j_residual"),
    ]
    _metric_bar_panel(
        axes[1],
        [a for a, _ in learned],
        [by_score[b] for _, b in learned],
        [METHOD_COLORS[b] for _, b in learned],
        "Learned action maps",
    )

    proxies = [
        ("IDM", "idm_reconstruction_error"),
        ("WAV", "wav_proxy"),
        ("wrong J", "wrong_j_residual"),
        ("smooth", "smoothness"),
    ]
    _metric_bar_panel(
        axes[2],
        [a for a, _ in proxies],
        [by_score[b] for _, b in proxies],
        [METHOD_COLORS[b] for _, b in proxies],
        "Hostile alternatives",
    )

    fig.suptitle("Failure prediction: clear signal without a dense ranking plot", x=0.02, y=1.02, ha="left", fontsize=10.5, fontweight="bold", color=PALETTE["black"])
    save_figure(fig, out_base)


def robustness_grid_plot(robust: pd.DataFrame, out_base: Path) -> None:
    set_style()
    envs = ["point_mass", "nonholonomic_car", "two_link_arm", "planar_pusher"]
    env_labels = ["point\nmass", "car", "arm", "pusher"]

    def value(env: str, variant: str, samples: int | None = None, noise: float | None = None) -> float:
        sub = robust[(robust["env"] == env) & (robust["variant"] == variant)]
        if samples is not None:
            sub = sub[sub["samples"] == samples]
        if noise is not None:
            sub = sub[np.isclose(sub["noise"], noise)]
        if sub.empty:
            return float("nan")
        return float(sub.sort_values(["samples", "noise"]).iloc[-1]["auroc"])

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65), sharey=True)
    fig.subplots_adjust(wspace=0.28)
    x = np.arange(len(envs))
    width = 0.34

    analytic = [value(env, "analytic") for env in envs]
    learned = [value(env, "learned_j", samples=512, noise=0.0) for env in envs]
    axes[0].bar(x - width / 2, analytic, width=width, color=PALETTE["blue_dark"], label="analytic", edgecolor="white", linewidth=0.6)
    bars = axes[0].bar(x + width / 2, learned, width=width, color=PALETTE["blue_mid"], label="learned J", edgecolor="white", linewidth=0.6)
    axes[0].set_title("Few-shot map recovers the certificate", fontsize=8.6)
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.7)

    clean = learned
    noisy = [value(env, "noisy_learned_j", samples=512, noise=0.65) for env in envs]
    axes[1].bar(x - width / 2, clean, width=width, color=PALETTE["blue_mid"], label="clean", edgecolor="white", linewidth=0.6)
    axes[1].bar(x + width / 2, noisy, width=width, color=PALETTE["blue_light"], label="noisy", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Noise stress keeps the story visible", fontsize=8.6)
    axes[1].legend(frameon=False, loc="lower left", fontsize=6.7)

    wrong = [value(env, "wrong_map") for env in envs]
    bars = axes[2].bar(x, wrong, width=0.56, color=[ENV_COLORS[e] for e in envs], edgecolor="white", linewidth=0.6)
    axes[2].set_title("Wrong maps are exposed", fontsize=8.6)
    for bar, val in zip(bars, wrong):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=6.8)

    for i, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(env_labels)
        ax.set_ylim(0.0, 1.06)
        _clean_numeric_axis(ax, y_label="AUROC" if i == 0 else None)
        ax.axhline(0.5, ls="--", lw=0.8, color=PALETTE["gray_mid"], zorder=0)
    fig.suptitle("Robustness audit: learned action maps hold until contact modes change", x=0.02, y=1.02, ha="left", fontsize=10.3, fontweight="bold", color=PALETTE["black"])
    save_figure(fig, out_base)


def visual_keypoint_plot(scores: pd.DataFrame, examples_npz: Path, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), gridspec_kw={"width_ratios": [0.92, 0.92, 1.42]})
    fig.subplots_adjust(wspace=0.45)
    data = np.load(examples_npz, allow_pickle=True) if examples_npz.exists() else None
    for ax, name in zip(axes[:2], ["nonholonomic_car", "planar_pusher"]):
        _card_background(ax)
        label = "A" if name == "nonholonomic_car" else "B"
        ax.text(0.02, 1.05, f"{label}  {name.replace('_', ' ')}", transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=PALETTE["black"])
        if data is None or name not in data["names"]:
            continue
        rep = data[f"{name}_rep"]
        pts = rep.reshape(len(rep), -1, 2)
        _set_equal_with_margin(ax, [pts[:, 0, :], pts[:, 1, :], pts[:, -1, :]], margin=0.18)
        for k in range(pts.shape[1]):
            color = PALETTE["blue_dark"] if k == 0 else PALETTE["blue_mid"] if k == 1 else PALETTE["orange"]
            ax.plot(pts[:, k, 0], pts[:, k, 1], color=color, lw=1.8, alpha=0.95)
            ax.scatter(pts[0, k, 0], pts[0, k, 1], s=18, color=PALETTE["black"], zorder=3)
            ax.scatter(pts[-1, k, 0], pts[-1, k, 1], s=26, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(0.04, 0.06, "black=start   color=end", transform=ax.transAxes, fontsize=6.6, color=PALETTE["soft_text"])
    ax = axes[2]
    _panel(ax, "C", "Visual representation scores")
    metric_rows = []
    for env, group in scores.groupby("env"):
        for score in ["visual_actionability", "learned_visual_actionability", "visual_smoothness"]:
            from actionability.metrics import safe_auc

            metric_rows.append({"env": env, "score": score, "auroc": safe_auc(group["failure"], group[score])})
    metric = pd.DataFrame(metric_rows)
    envs = ["nonholonomic_car", "planar_pusher"]
    score_order = [
        ("analytic", "visual_actionability", PALETTE["blue_dark"]),
        ("learned", "learned_visual_actionability", PALETTE["blue_mid"]),
        ("smooth", "visual_smoothness", PALETTE["gray_mid"]),
    ]
    x = np.arange(len(envs))
    width = 0.22
    for i, (label, score, color) in enumerate(score_order):
        vals = [
            float(metric[(metric["env"] == env) & (metric["score"] == score)]["auroc"].iloc[0])
            for env in envs
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=6.7, color=PALETTE["black"])
    ax.set_xticks(x)
    ax.set_xticklabels(["car", "pusher"])
    ax.set_ylim(0.0, 1.07)
    _clean_numeric_axis(ax, y_label="AUROC")
    ax.axhline(0.5, ls="--", lw=0.8, color=PALETTE["gray_mid"], zorder=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), columnspacing=0.9, handlelength=1.1)
    save_figure(fig, out_base)


def environment_grid(examples: dict, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.1, 2.25))
    fig.subplots_adjust(wspace=0.18)
    reasons = {
        "point_mass": "positive control",
        "nonholonomic_car": "sideways future",
        "two_link_arm": "limited tangent",
        "planar_pusher": "before contact",
    }
    for ax, (name, data) in zip(axes, examples.items()):
        _card_background(ax)
        traj = data["candidate"]
        rollout = data["rollout"]
        _set_equal_with_margin(ax, [traj, rollout], margin=0.16)
        ax.plot(traj[:, 0], traj[:, 1], color=PALETTE["rose"], lw=2.1, label="imagined")
        ax.plot(rollout[:, 0], rollout[:, 1], color=PALETTE["blue_dark"], lw=2.1, label="executed")
        ax.scatter(traj[0, 0], traj[0, 1], s=26, color=PALETTE["black"], zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=34, color=PALETTE["orange"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(0.05, 0.91, name.replace("_", " "), transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=PALETTE["black"])
        ax.text(0.05, 0.79, reasons.get(name, ""), transform=ax.transAxes, fontsize=6.8, color=PALETTE["soft_text"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, handlelength=1.7)
    save_figure(fig, out_base)


def failure_prediction_plot(scores: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    fig.subplots_adjust(wspace=0.30)
    colors = {0: PALETTE["blue_mid"], 1: PALETTE["rose"]}
    for fail, group in scores.groupby("failure"):
        axes[0].scatter(
            group["actionability_residual"],
            group["rollout_error"],
            s=12,
            alpha=0.55,
            color=colors[int(fail)],
            label="failure" if fail else "success",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("actionability residual")
    axes[0].set_ylabel("execution error")
    axes[0].legend(frameon=False)
    axes[0].set_title("Residual predicts execution gap", fontsize=8.8)
    _clean_numeric_axis(axes[0])
    for env, group in scores.groupby("env"):
        order = np.argsort(group["actionability_residual"].to_numpy())
        x = np.linspace(0, 1, len(group))
        y = group.iloc[order]["failure"].rolling(max(4, len(group) // 12), min_periods=1).mean()
        axes[1].plot(x, y, lw=2.0, label=env.replace("_", " "), color=ENV_COLORS.get(env, PALETTE["blue_mid"]))
    axes[1].set_xlabel("residual quantile")
    axes[1].set_ylabel("empirical failure rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Failure rate rises with residual", fontsize=8.8)
    axes[1].legend(frameon=False, ncol=1, fontsize=6.8, loc="upper left")
    _clean_numeric_axis(axes[1])
    save_figure(fig, out_base)


def repair_plot(traj_npz: Path, out_base: Path) -> None:
    set_style()
    data = np.load(traj_npz, allow_pickle=True)
    names = list(data["names"])
    fig, axes = plt.subplots(1, len(names), figsize=(8.3, 2.45))
    fig.subplots_adjust(wspace=0.18)
    for ax, name in zip(axes, names):
        _card_background(ax)
        before = data[f"{name}_before"]
        after = data[f"{name}_after"]
        rollout_before = data[f"{name}_rollout_before"]
        rollout_after = data[f"{name}_rollout_after"]
        _set_equal_with_margin(ax, [before, after, rollout_before, rollout_after], margin=0.16)
        before_gap = float(np.mean(np.linalg.norm(before[:, :2] - rollout_before[:, :2], axis=1)))
        after_gap = float(np.mean(np.linalg.norm(after[:, :2] - rollout_after[:, :2], axis=1)))
        drop = max(0.0, 1.0 - after_gap / max(before_gap, 1e-9))
        ax.plot(before[:, 0], before[:, 1], color=PALETTE["rose"], lw=1.5, alpha=0.65, label="before dream")
        ax.plot(rollout_before[:, 0], rollout_before[:, 1], "--", color=PALETTE["rose"], lw=1.1, alpha=0.65, label="before rollout")
        ax.plot(after[:, 0], after[:, 1], color=PALETTE["blue_dark"], lw=2.2, label="repaired dream")
        ax.plot(rollout_after[:, 0], rollout_after[:, 1], "--", color=PALETTE["blue_mid"], lw=1.4, label="repaired rollout")
        ax.scatter(after[0, 0], after[0, 1], s=24, color=PALETTE["black"], zorder=4)
        ax.scatter(after[-1, 0], after[-1, 1], s=34, color=PALETTE["orange"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(0.05, 0.91, name.replace("_", " "), transform=ax.transAxes, fontsize=8.3, fontweight="bold", color=PALETTE["black"])
        ax.text(
            0.05,
            0.07,
            f"execution gap down {100 * drop:.0f}%",
            transform=ax.transAxes,
            fontsize=6.7,
            color=PALETTE["blue_dark"],
            fontweight="bold",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, handlelength=1.8, columnspacing=0.9)
    save_figure(fig, out_base)


def embodiment_heatmap(df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    envs = list(df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(8.25, 2.35), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.18)
    vmax = np.percentile(df["residual"], 96)
    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        xs = np.sort(sub["dx"].unique())
        ys = np.sort(sub["dy"].unique())
        grid = sub.pivot(index="dy", columns="dx", values="residual").loc[ys, xs].to_numpy()
        im = ax.imshow(grid, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap=RESIDUAL_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.contour(xs, ys, grid, levels=[max(vmax * 0.35, 1e-6)], colors=["#f7f1df"], linewidths=0.7, alpha=0.78)
        ax.axhline(0, color="white", lw=0.7, alpha=0.85)
        ax.axvline(0, color="white", lw=0.7, alpha=0.85)
        ax.set_title(env.replace("_", " "), fontsize=8.9, color=PALETTE["black"])
        ax.set_xlabel("dx")
        ax.set_aspect("equal")
        ax.tick_params(length=2.5, width=0.7, colors=PALETTE["soft_text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["gray_mid"])
            spine.set_linewidth(0.7)
    axes[0].set_ylabel("dy")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("actionability residual", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Same imagined displacement, different embodiment certificate", x=0.02, y=1.02, ha="left", fontsize=10.3, fontweight="bold", color=PALETTE["black"])
    save_figure(fig, out_base)


def learned_j_curve(df: pd.DataFrame, out_base: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    for env, group in df.groupby("env"):
        group = group.sort_values("samples")
        color = ENV_COLORS.get(env, PALETTE["blue_mid"])
        ax.plot(group["samples"], group["j_error"], marker="o", lw=2.0, ms=4.5, label=env.replace("_", " "), color=color)
        ax.text(group["samples"].iloc[-1] * 1.05, group["j_error"].iloc[-1], env.replace("_", " "), color=color, fontsize=7.2, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training transitions")
    ax.set_ylabel("relative Jacobian error")
    ax.set_title("Small local action-map models improve with data", fontsize=9.5, color=PALETTE["black"])
    ax.set_xlim(df["samples"].min() * 0.9, df["samples"].max() * 1.65)
    _clean_numeric_axis(ax)
    save_figure(fig, out_base)
