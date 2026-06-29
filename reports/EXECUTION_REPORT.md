# Execution Report

Date: 2026-06-29
Machine timezone from thread context: Asia/Shanghai

## Status

Upgraded research package is complete locally on branch `learned-j-bulletproof` through full experiments, figures, paper rewrite, reports, final command-gate rerun, and PDF render check. Commit and GitHub push are performed after this final report edit.

## Repository

Local path: `C:\Users\wangz\actionability-energy`

GitHub URL: `https://github.com/Jason-Wang313/actionability-energy`

Working branch: `learned-j-bulletproof`

## Paper PDF

Compiled PDF target: `C:\Users\wangz\actionability-energy\paper\main.pdf`

PDF render check:

- Rendered page 1, page 2, and page 6 with PyMuPDF under `reports/rendered/`.
- Sampled pages were nonblank and visually coherent.

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

- `pytest -q` PASS, 8 tests.
- `python experiments/run_all.py --quick` PASS.
- `python experiments/run_all.py --full` PASS.
- `bash scripts/make_plots.sh` PASS.
- `bash scripts/compile_paper.sh` PASS.
- `python experiments/run_all.py --plots-only` PASS during figure QA.

Compilation note: `latexmk` exists but MiKTeX cannot run it because Perl is missing. The compile script falls back to direct `pdflatex`/`bibtex`. The final log has no unresolved citations or references.

## Upgraded Experiment Status

Full run manifest:

- Mode: full.
- Candidate count: 360.
- Figures expected: 9.

Main full-run metrics from `results/processed/failure_prediction_metrics.csv`:

- learned classifier: AUROC 1.000, AUPRC 1.000. This is reported as synthetic-distribution pattern fitting, not mechanism.
- analytic actionability residual: AUROC 0.989, AUPRC 0.989.
- learned-J residual: AUROC 0.913, AUPRC 0.883.
- noisy learned-J residual: AUROC 0.884, AUPRC 0.858.
- inverse-dynamics reconstruction error: AUROC 0.883, AUPRC 0.883.
- WAV-style proxy: AUROC 0.799, AUPRC 0.772.
- wrong-map residual: AUROC 0.620, AUPRC 0.553.
- smoothness: AUROC 0.426, AUPRC 0.426.

Robustness audit at 512 transitions:

- point_mass learned-J AUROC 1.000.
- nonholonomic_car learned-J AUROC 0.985.
- two_link_arm learned-J AUROC 0.948.
- planar_pusher learned-J AUROC 0.604, exposing contact-mode discontinuity.

Visual/keypoint futures:

- car analytic visual actionability AUROC 1.000; learned visual actionability AUROC 0.961.
- pusher analytic visual actionability AUROC 1.000; learned visual actionability AUROC 0.509, exposing contact-mode mismatch in visual representation space.

Repair diagnostic cases:

- Full actionability energy repairs point_mass, nonholonomic_car, two_link_arm, and planar_pusher in the analytic setting.
- Upgraded repair table also includes learned-J, noisy learned-J, and wrong-map variants.

Generated upgraded figures:

- `fig1_concept_actionability`: main diagnostic-interface story.
- `fig2_learned_j_interface`: learned map as inverse action, diagnosis, and repair.
- `fig3_hostile_baselines`: hostile baseline comparison.
- `fig4_robustness_audit`: sample/noise robustness grid.
- `fig5_repair_before_after`: repair rollouts.
- `fig6_embodiment_swap`: same dream, different body.
- `fig7_learned_jacobian`: learned-J sample curve.
- `fig8_visual_keypoint`: CPU visual/keypoint futures.

## Unresolved Issues

- No real robot validation.
- No PushT/ManiSkill optional experiment.
- Repair solver is projected and CPU-practical, not a full nonlinear optimizer.
- Large-video-model future generation is represented by controlled proxy generators.
- Learned pusher/contact results are weak; this is the main scientific failure mode and motivates active contact probing or mode-aware Jacobian fields.
- Figures are substantially improved over v0 but could still benefit from a Figma/Illustrator pass against Yilun Du-style visual references.

## Submission Readiness

Score: 7.2 / 10.

The upgraded package is a credible Lester/NJF/VERA-aligned mechanism paper: learned local action maps are central, hostile baselines are included, robustness audits are explicit, and visual/keypoint futures reduce the pure-state-space feel. It is still not a fully bulletproof ICLR-main submission because validation remains synthetic and the learned contact result exposes a real open problem.
