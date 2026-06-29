from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actionability.metrics import calibration_bins
from actionability.plotting import (
    concept_figure,
    embodiment_heatmap,
    environment_grid,
    failure_prediction_plot,
    geometry_projection,
    learned_j_curve,
    repair_plot,
)
from experiments.exp_embodiment_swap import run_embodiment_swap
from experiments.exp_failure_prediction import run_failure_prediction
from experiments.exp_learned_jacobian import run_learned_jacobian
from experiments.exp_repair import run_repair_experiment


RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGS = ROOT / "results" / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"


def _write_table_for_latex(metrics: pd.DataFrame, repair: pd.DataFrame) -> None:
    table_dir = ROOT / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    act = metrics[metrics["score"] == "actionability_residual"].copy()
    act["env"] = act["env"].str.replace("_", " ", regex=False)
    with (table_dir / "failure_prediction.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccc}\\toprule\n")
        f.write("Environment & AUROC & AUPRC & Acc.\\\\\\midrule\n")
        for _, row in act.iterrows():
            f.write(f"{row['env']} & {row['auroc']:.3f} & {row['auprc']:.3f} & {row['threshold_accuracy']:.3f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
    pivot = repair.pivot_table(index="env", columns="variant", values="success", aggfunc="mean").fillna(0)
    with (table_dir / "repair_success.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcc}\\toprule\n")
        f.write("Environment & Candidate & Full energy\\\\\\midrule\n")
        for env, row in pivot.iterrows():
            f.write(f"{env.replace('_', ' ')} & {row.get('candidate', 0):.0f} & {row.get('full_energy', 0):.0f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


def _save_repair_npz(payload: dict, path: Path) -> None:
    arrays = {"names": np.asarray(list(payload.keys()), dtype=object)}
    for name, parts in payload.items():
        for key, value in parts.items():
            arrays[f"{name}_{key}"] = value
    np.savez(path, **arrays)


def make_plots() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    concept_figure(FIGS / "fig1_concept_actionability")
    geometry_projection(FIGS / "fig2_geometry_projection")
    scores_path = RAW / "failure_prediction_scores.csv"
    repair_npz = RAW / "repair_trajectories.npz"
    swap_path = RAW / "embodiment_swap.csv"
    learned_path = RAW / "learned_jacobian.csv"
    examples_path = RAW / "environment_examples.npz"
    if scores_path.exists():
        failure_prediction_plot(pd.read_csv(scores_path), FIGS / "fig4_failure_prediction")
    if repair_npz.exists():
        repair_plot(repair_npz, FIGS / "fig5_repair_before_after")
    if swap_path.exists():
        embodiment_heatmap(pd.read_csv(swap_path), FIGS / "fig6_embodiment_swap")
    if learned_path.exists():
        learned_j_curve(pd.read_csv(learned_path), FIGS / "fig7_learned_jacobian")
    if examples_path.exists():
        data = np.load(examples_path, allow_pickle=True)
        examples = {}
        for name in data["names"]:
            examples[str(name)] = {
                "candidate": data[f"{name}_candidate"],
                "rollout": data[f"{name}_rollout"],
            }
        environment_grid(examples, FIGS / "fig3_environment_grid")
    for fig in FIGS.glob("*.pdf"):
        shutil.copy2(fig, PAPER_FIGS / fig.name)
    for fig in FIGS.glob("*.png"):
        shutil.copy2(fig, PAPER_FIGS / fig.name)


def run(mode: str, plots_only: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    quick = mode == "quick"
    if not plots_only:
        n = 28 if quick else 90
        T = 14 if quick else 22
        scores, metrics, examples = run_failure_prediction(n=n, T=T, seed=42 if quick else 84)
        scores.to_csv(RAW / "failure_prediction_scores.csv", index=False)
        metrics.to_csv(PROCESSED / "failure_prediction_metrics.csv", index=False)
        example_arrays = {"names": np.asarray(list(examples.keys()), dtype=object)}
        for name, item in examples.items():
            example_arrays[f"{name}_candidate"] = item["candidate"]
            example_arrays[f"{name}_rollout"] = item["rollout"]
        np.savez(RAW / "environment_examples.npz", **example_arrays)
        cal_rows = []
        for env, group in scores.groupby("env"):
            for row in calibration_bins(group["failure"], group["actionability_residual"]):
                row["env"] = env
                cal_rows.append(row)
        pd.DataFrame(cal_rows).to_csv(PROCESSED / "calibration_bins.csv", index=False)
        repair, repair_payload = run_repair_experiment(T=T, seed=77 if quick else 99)
        repair.to_csv(PROCESSED / "repair_results.csv", index=False)
        _save_repair_npz(repair_payload, RAW / "repair_trajectories.npz")
        swap = run_embodiment_swap(grid=23 if quick else 35)
        swap.to_csv(RAW / "embodiment_swap.csv", index=False)
        learned = run_learned_jacobian(seed=123, quick=quick)
        learned.to_csv(RAW / "learned_jacobian.csv", index=False)
        _write_table_for_latex(metrics, repair)
        manifest = {
            "mode": mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidate_count": int(len(scores)),
            "figures_expected": 7,
        }
        (PROCESSED / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_plots()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    run(mode, plots_only=args.plots_only)


if __name__ == "__main__":
    main()
