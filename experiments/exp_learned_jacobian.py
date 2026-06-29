from __future__ import annotations

import numpy as np
import pandas as pd

from actionability.jacobians import LocalLinearJacobian, collect_rollout_jacobian_data
from envs import make_envs


def run_learned_jacobian(seed: int, quick: bool = False) -> pd.DataFrame:
    sample_sizes = [24, 64, 128] if quick else [32, 64, 128, 256, 512]
    rows = []
    for env_idx, env in enumerate(make_envs()):
        z_train, u_train, d_train = collect_rollout_jacobian_data(env, max(sample_sizes), seed + env_idx)
        rng = np.random.default_rng(seed + 1000 + env_idx)
        z_test = np.asarray([env.sample_state(rng) for _ in range(80 if quick else 160)])
        for n in sample_sizes:
            model = LocalLinearJacobian(k=min(48, n), length_scale=0.65).fit(z_train[:n], u_train[:n], d_train[:n])
            errs = []
            norms = []
            for z in z_test:
                J_true = env.jacobian(z)
                J_hat = model.predict(z)
                errs.append(float(np.linalg.norm(J_hat - J_true)))
                norms.append(float(np.linalg.norm(J_true)) + 1e-8)
            rows.append(
                {
                    "env": env.name,
                    "samples": int(n),
                    "j_error": float(np.mean(errs) / np.mean(norms)),
                    "abs_j_error": float(np.mean(errs)),
                }
            )
    return pd.DataFrame(rows)
