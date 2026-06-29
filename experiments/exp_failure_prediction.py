from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from actionability.jacobians import InverseDynamicsKNN, LocalLinearJacobian, collect_rollout_jacobian_data, recommended_local_j_params
from actionability.metrics import safe_auc, safe_auprc, threshold_accuracy
from envs import make_envs
from experiments.common import score_candidates


BASELINES = [
    "actionability_residual",
    "learned_j_residual",
    "noisy_j_residual",
    "wrong_j_residual",
    "idm_reconstruction_error",
    "wav_proxy",
    "learned_classifier",
    "mean_residual",
    "action_norm",
    "smoothness",
    "passive_nll",
    "path_length",
    "random_score",
]


CLASSIFIER_FEATURES = [
    "smoothness",
    "path_length",
    "passive_nll",
    "action_norm",
    "idm_reconstruction_error",
    "wav_proxy",
]


def _train_local_models(env, seed: int, quick: bool):
    n = 160 if quick else 700
    z, u, d = collect_rollout_jacobian_data(env, n, seed=seed, action_scale=0.35)
    z_inv, u_inv, d_inv = collect_rollout_jacobian_data(env, n, seed=seed + 900, action_scale=1.0)
    k, length_scale = recommended_local_j_params(env, n)
    learned_j = LocalLinearJacobian(k=k, length_scale=length_scale).fit(z, u, d)
    inverse = InverseDynamicsKNN(k=24 if quick else 40).fit(z_inv, u_inv, d_inv)
    return learned_j, inverse


def _add_classifier_scores(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test = test.copy()
    y = train["failure"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        test["learned_classifier"] = float(y[0]) if len(y) else 0.5
        return test
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        min_samples_leaf=4,
        random_state=17,
        class_weight="balanced_subsample",
    )
    clf.fit(train[CLASSIFIER_FEATURES].fillna(0.0), y)
    test["learned_classifier"] = clf.predict_proba(test[CLASSIFIER_FEATURES].fillna(0.0))[:, 1]
    return test


def run_failure_prediction(n: int, T: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    examples = {}
    quick = n <= 30
    for offset, env in enumerate(make_envs()):
        learned_j, inverse = _train_local_models(env, seed + 100 * offset, quick=quick)
        train_scores, _ = score_candidates(env, max(24, n // 2), T, seed + 1000 + 17 * offset, learned_j, inverse)
        scores, example = score_candidates(env, n, T, seed + 17 * offset, learned_j, inverse)
        scores = _add_classifier_scores(train_scores, scores)
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
    for score_name in BASELINES:
        rows.append(
            {
                "env": "all",
                "score": score_name,
                "auroc": safe_auc(all_scores["failure"], all_scores[score_name]),
                "auprc": safe_auprc(all_scores["failure"], all_scores[score_name]),
                "threshold_accuracy": threshold_accuracy(all_scores["failure"], all_scores[score_name]),
            }
        )
    return all_scores, pd.DataFrame(rows), examples
