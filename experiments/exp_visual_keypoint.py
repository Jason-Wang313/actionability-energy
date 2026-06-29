from __future__ import annotations

import numpy as np
import pandas as pd

from actionability.jacobians import LocalLinearJacobian, recommended_local_j_params
from actionability.metrics import safe_auc, safe_auprc
from actionability.residuals import solve_actionability
from envs import NonholonomicCarEnv, PlanarPusherEnv
from experiments.common import candidate_trajectories, rollout_with_recovered_controls, smoothness_score


def car_keypoints(state: np.ndarray) -> np.ndarray:
    x, y, theta = np.asarray(state, dtype=float)
    nose = np.array([x + 0.32 * np.cos(theta), y + 0.32 * np.sin(theta)])
    side = np.array([x - 0.16 * np.sin(theta), y + 0.16 * np.cos(theta)])
    return np.r_[x, y, nose, side]


def pusher_keypoints(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float)
    pusher = s[:2]
    obj = s[2:4]
    return np.r_[pusher, obj, obj + np.array([0.18, 0.0]), obj + np.array([0.0, 0.18])]


def finite_diff_rep_jacobian(env, rep_fn, state: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    base = rep_fn(state)
    cols = []
    for i in range(env.action_dim):
        u = np.zeros(env.action_dim)
        u[i] = eps
        cols.append((rep_fn(env.step(state, u)) - base) / eps)
    return np.stack(cols, axis=1)


def _visual_residual(env, rep_traj: np.ndarray, state_traj: np.ndarray, rep_j_fn) -> float:
    low, high = env.action_bounds
    total = 0.0
    for t in range(len(rep_traj) - 1):
        total += solve_actionability(rep_traj[t + 1] - rep_traj[t], rep_j_fn(state_traj[t], rep_traj[t]), bounds=(low, high)).residual
    return float(total)


def _collect_visual_j_data(env, rep_fn, n: int, seed: int, action_scale: float = 0.35):
    rng = np.random.default_rng(seed)
    z_rows, u_rows, d_rows = [], [], []
    low, high = env.action_bounds
    for _ in range(n):
        state = env.sample_state(rng)
        rep = rep_fn(state)
        u = rng.uniform(low * action_scale, high * action_scale)
        nxt = env.step(state, u)
        z_rows.append(rep)
        u_rows.append(u)
        d_rows.append(rep_fn(nxt) - rep)
    return np.asarray(z_rows), np.asarray(u_rows), np.asarray(d_rows)


def run_visual_keypoint(seed: int, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    env_specs = [(NonholonomicCarEnv(), car_keypoints), (PlanarPusherEnv(), pusher_keypoints)]
    n = 36 if quick else 90
    T = 14 if quick else 22
    rows = []
    examples = {}
    for env_idx, (env, rep_fn) in enumerate(env_specs):
        z, u, d = _collect_visual_j_data(env, rep_fn, 180 if quick else 700, seed + 50 * env_idx, action_scale=0.35)
        k, length_scale = recommended_local_j_params(env, len(z))
        learned_visual_j = LocalLinearJacobian(k=k, length_scale=length_scale).fit(z, u, d)
        for cid, item in enumerate(candidate_trajectories(env, n, T, seed + 200 + env_idx)):
            state_traj = item["trajectory"]
            rep_traj = np.asarray([rep_fn(s) for s in state_traj])
            label = int(not rollout_with_recovered_controls(env, state_traj)["success"])
            analytic = _visual_residual(env, rep_traj, state_traj, lambda state, rep: finite_diff_rep_jacobian(env, rep_fn, state))
            learned = _visual_residual(env, rep_traj, state_traj, lambda state, rep: learned_visual_j.predict(rep))
            rows.append(
                {
                    "env": env.name,
                    "candidate_id": cid,
                    "failure": label,
                    "visual_actionability": analytic,
                    "learned_visual_actionability": learned,
                    "visual_smoothness": smoothness_score(rep_traj),
                    "visual_path_length": float(np.sum(np.linalg.norm(np.diff(rep_traj.reshape(len(rep_traj), -1, 2)[:, 1], axis=0), axis=1))),
                }
            )
            if env.name not in examples and label == 1:
                examples[env.name] = {"state": state_traj, "rep": rep_traj}
    scores = pd.DataFrame(rows)
    metrics = []
    for env, group in scores.groupby("env"):
        for score in ["visual_actionability", "learned_visual_actionability", "visual_smoothness", "visual_path_length"]:
            metrics.append(
                {
                    "env": env,
                    "score": score,
                    "auroc": safe_auc(group["failure"], group[score]),
                    "auprc": safe_auprc(group["failure"], group[score]),
                }
            )
    return scores, pd.DataFrame(metrics), examples
