from __future__ import annotations

import numpy as np
import pandas as pd

from actionability.jacobians import LocalLinearJacobian, collect_rollout_jacobian_data, corrupted_jacobian, noisy_jacobian, recommended_local_j_params
from actionability.metrics import safe_auc, safe_auprc
from envs import make_envs
from experiments.common import candidate_trajectories, residual_score, rollout_with_recovered_controls


def run_robustness(seed: int, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_sizes = [24, 64, 128] if quick else [32, 64, 128, 256, 512]
    noise_levels = [0.0, 0.15, 0.35, 0.65]
    all_rows = []
    metric_rows = []
    n_candidates = 32 if quick else 80
    T = 14 if quick else 22
    for env_idx, env in enumerate(make_envs()):
        candidates = candidate_trajectories(env, n_candidates, T, seed + 31 * env_idx)
        labels = []
        for item in candidates:
            labels.append(int(not rollout_with_recovered_controls(env, item["trajectory"])["success"]))
        z_all, u_all, d_all = collect_rollout_jacobian_data(env, max(sample_sizes), seed + 100 * env_idx, action_scale=0.35)
        analytic_scores = [residual_score(env, item["trajectory"]) for item in candidates]
        metric_rows.append(
            {
                "env": env.name,
                "variant": "analytic",
                "samples": 0,
                "noise": 0.0,
                "auroc": safe_auc(labels, analytic_scores),
                "auprc": safe_auprc(labels, analytic_scores),
            }
        )
        wrong_scores = [residual_score(env, item["trajectory"], jacobian_fn=lambda z, env=env: corrupted_jacobian(env, z)) for item in candidates]
        metric_rows.append(
            {
                "env": env.name,
                "variant": "wrong_map",
                "samples": 0,
                "noise": 0.0,
                "auroc": safe_auc(labels, wrong_scores),
                "auprc": safe_auprc(labels, wrong_scores),
            }
        )
        for n in sample_sizes:
            k, length_scale = recommended_local_j_params(env, n)
            model = LocalLinearJacobian(k=k, length_scale=length_scale).fit(z_all[:n], u_all[:n], d_all[:n])
            for noise in noise_levels:
                rng = np.random.default_rng(seed + env_idx * 1000 + n + int(noise * 100))

                def j_fn(z, model=model, noise=noise, rng=rng):
                    J = model.predict(z)
                    if noise > 0:
                        J = noisy_jacobian(J, noise, rng)
                    return J

                scores = [residual_score(env, item["trajectory"], jacobian_fn=j_fn) for item in candidates]
                metric_rows.append(
                    {
                        "env": env.name,
                        "variant": "learned_j" if noise == 0 else "noisy_learned_j",
                        "samples": n,
                        "noise": noise,
                        "auroc": safe_auc(labels, scores),
                        "auprc": safe_auprc(labels, scores),
                    }
                )
                for cid, score, label in zip(range(len(candidates)), scores, labels):
                    all_rows.append(
                        {
                            "env": env.name,
                            "candidate_id": cid,
                            "variant": "learned_j" if noise == 0 else "noisy_learned_j",
                            "samples": n,
                            "noise": noise,
                            "score": float(score),
                            "failure": int(label),
                        }
                    )
    return pd.DataFrame(all_rows), pd.DataFrame(metric_rows)
