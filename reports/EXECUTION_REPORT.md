# Execution Report

Date: 2026-06-29
Machine timezone from thread context: Asia/Shanghai

## Status

PARTIAL-to-DONE local package: code, tests, experiments, figures, paper source, compiled PDF, and reports are complete locally. GitHub push status is pending at the time this report was first written.

## Repository

Local path: `C:\Users\wangz\actionability-energy`

## Paper PDF

Compiled PDF: `C:\Users\wangz\actionability-energy\paper\main.pdf`

PDF render check:

- Rendered page 1 with PyMuPDF to `reports/rendered/main_page1.png`.
- Rendered page 4 with PyMuPDF to `reports/rendered/main_page4.png`.
- Both sampled pages rendered nonblank and visually coherent.

## Hardware

- Machine: Lenovo 20W1A15QCD.
- CPU: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz.
- Cores/logical processors: 4 / 8.
- RAM: 16.9 GB reported by Windows.
- GPU: not used.

## Dependency Versions

- Python: 3.10.11.
- Platform: Windows-10-10.0.26200-SP0.
- numpy: 1.26.4.
- scipy: 1.15.3.
- matplotlib: 3.10.8.
- pandas: 2.3.3.
- scikit-learn: 1.7.2.
- pytest: 9.0.2.
- LaTeX: MiKTeX pdfTeX 1.40.28.

## Commands Run

- `pytest -q` PASS.
- `python experiments/run_all.py --quick` PASS.
- `python experiments/run_all.py --full` PASS.
- `bash scripts/make_plots.sh` PASS.
- `bash scripts/compile_paper.sh` PASS after fallback.

Compilation note: `latexmk` exists but MiKTeX cannot run it because Perl is missing. The script now falls back to direct `pdflatex`/`bibtex`, which produced `paper/main.pdf`.

## Experiment Status

Full run manifest:

- Mode: full.
- Candidate count: 360.
- Figures expected: 7.
- Figures generated: 7 PDF + 7 PNG.

Main actionability residual failure-prediction results from `results/processed/failure_prediction_metrics.csv`:

- nonholonomic_car: AUROC 1.000, AUPRC 1.000.
- planar_pusher: AUROC 1.000, AUPRC 1.000.
- point_mass: AUROC 1.000, AUPRC 1.000.
- two_link_arm: AUROC 0.990, AUPRC 0.966.
- all: AUROC 0.989, AUPRC 0.989.

Repair diagnostic cases from `results/processed/repair_results.csv`:

- Full actionability energy repairs point_mass, nonholonomic_car, two_link_arm, and planar_pusher to successful recovered-control rollouts.
- Smoothness-only variants help some geometry but do not reliably repair non-executability.

Generated figures:

- `fig1_concept_actionability`
- `fig2_geometry_projection`
- `fig3_environment_grid`
- `fig4_failure_prediction`
- `fig5_repair_before_after`
- `fig6_embodiment_swap`
- `fig7_learned_jacobian`

## Unresolved Issues

- No real robot validation.
- No PushT/ManiSkill optional experiment.
- Repair solver is projected and CPU-practical, not a full nonlinear optimizer.
- Large-video-model future generation is represented by controlled proxy generators.
- Figure quality is solid but would benefit from a design pass before a serious ICLR submission.

## Submission Readiness

Score: 6.5 / 10.

The package is coherent and reproducible, with a clear novelty boundary and complete local artifacts. It is not yet a true ICLR-main submission because validation is toy-only and the repair solver is not as strong as the framing could support. As a bridge paper or workshop-quality seed for Jason's WAM/actionability agenda, it is in good shape.
