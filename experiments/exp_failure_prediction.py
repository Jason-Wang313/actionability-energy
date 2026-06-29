from __future__ import annotations

import pandas as pd

from actionability.metrics import safe_auc, safe_auprc, threshold_accuracy
from envs import make_envs
from experiments.common import score_candidates


BASELINES = [
    "actionability_residual",
    "mean_residual",
    "action_norm",
    "smoothness",
    "passive_nll",
    "path_length",
    "random_score",
]


def run_failure_prediction(n: int, T: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    examples = {}
    for offset, env in enumerate(make_envs()):
        scores, example = score_candidates(env, n, T, seed + 17 * offset)
        frames.append(scores)
        examples[env.name] = example
    all_scores = pd.concat(frames, ignore_index=True)
    rows = []
    for env_name, group in all_scores.groupby("env"):
        for score_name in BASELINES:
            rows.append(
                {
                    "env": env_name,
                    "score": score_name,
                    "auroc": safe_auc(group["failure"], group[score_name]),
                    "auprc": safe_auprc(group["failure"], group[score_name]),
                    "threshold_accuracy": threshold_accuracy(group["failure"], group[score_name]),
                }
            )
    rows.append(
        {
            "env": "all",
            "score": "actionability_residual",
            "auroc": safe_auc(all_scores["failure"], all_scores["actionability_residual"]),
            "auprc": safe_auprc(all_scores["failure"], all_scores["actionability_residual"]),
            "threshold_accuracy": threshold_accuracy(all_scores["failure"], all_scores["actionability_residual"]),
        }
    )
    return all_scores, pd.DataFrame(rows), examples
