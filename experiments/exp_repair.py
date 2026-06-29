from __future__ import annotations

from experiments.common import run_repair_suite


def run_repair_experiment(T: int, seed: int):
    return run_repair_suite(T=T, seed=seed)
