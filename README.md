# Actionability Energy

CPU-first research package for **Actionability Energy: A Tangent-Space Test for Controllable Visual Futures**.

The project tests a simple claim: a generated future is not useful for control just because it is visually plausible or goal-reaching. It should also be locally reachable under the embodiment's action map. The package implements actionability residuals, toy embodiment benchmarks, repair by compositional trajectory inference, plots, and an ICLR-style paper draft.

## Quick Start

```bash
python -m pip install -r requirements.txt
pytest -q
python experiments/run_all.py --quick
bash scripts/make_plots.sh
bash scripts/compile_paper.sh
```

The main CPU run is:

```bash
python experiments/run_all.py --full
```

## Outputs

- Raw results: `results/raw/`
- Processed tables: `results/processed/`
- Figures: `results/figures/` and `paper/figures/`
- Paper source: `paper/main.tex`
- Compiled paper: `paper/main.pdf` after `bash scripts/compile_paper.sh`
- Execution and novelty reports: `reports/`

## Scope

This is a controlled, low-compute benchmark. It does not claim real-robot validation, large video-model fine-tuning, or state-of-the-art robot policy performance. The contribution is the tangent-space residual, its diagnostic benchmark, and its use as a composable repair factor.

## Main Commands Used For Verification

```bash
pytest -q
python experiments/run_all.py --quick
python experiments/run_all.py --full
bash scripts/make_plots.sh
bash scripts/compile_paper.sh
```
