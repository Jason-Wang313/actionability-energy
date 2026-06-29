from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actionability.residuals import solve_actionability
from actionability.repair import RepairConfig, repair_trajectory
from envs import NonholonomicCarEnv, PlanarPusherEnv, PointMassEnv, TwoLinkArmEnv, make_envs


def interpolate(start: np.ndarray, goal: np.ndarray, T: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, T + 1)[:, None]
    return (1.0 - alpha) * start[None, :] + alpha * goal[None, :]


def smoothness_score(traj: np.ndarray) -> float:
    if len(traj) < 3:
        return 0.0
    second = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    return float(np.sum(second * second))


def rollout_with_recovered_controls(env, traj: np.ndarray) -> dict:
    low, high = env.action_bounds
    z = np.asarray(traj[0], dtype=float).copy()
    rollout = [z.copy()]
    residuals, controls, action_norms = [], [], []
    for t in range(len(traj) - 1):
        delta = traj[t + 1] - traj[t]
        out = solve_actionability(delta, env.jacobian(traj[t]), bounds=(low, high), lam=1e-4)
        controls.append(out.u)
        residuals.append(out.residual)
        action_norms.append(float(np.linalg.norm(out.u)))
        z = env.step(z, out.u)
        rollout.append(z.copy())
    rollout = np.asarray(rollout)
    residuals = np.asarray(residuals)
    controls = np.asarray(controls)
    state_err = np.linalg.norm(rollout - traj, axis=1)
    xy_err = np.linalg.norm(env.plot_xy(rollout) - env.plot_xy(traj), axis=1)
    return {
        "rollout": rollout,
        "controls": controls,
        "residuals": residuals,
        "actionability_residual": float(residuals.sum()),
        "mean_residual": float(residuals.mean()),
        "action_norm": float(np.mean(action_norms)),
        "rollout_error": float(np.mean(state_err) + state_err[-1]),
        "xy_error": float(np.mean(xy_err) + xy_err[-1]),
        "success": bool(xy_err[-1] < env.success_threshold and np.mean(xy_err) < env.success_threshold),
    }


def _scripted_rollout(env, start: np.ndarray, controls: np.ndarray) -> np.ndarray:
    traj = [np.asarray(start, dtype=float)]
    z = traj[0].copy()
    for u in controls:
        z = env.step(z, u)
        traj.append(z.copy())
    return np.asarray(traj)


def candidate_trajectories(env, n: int, T: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = i % 5
        start = env.default_start()
        if kind == 0:
            controls = rng.uniform(env.action_bounds[0] * 0.65, env.action_bounds[1] * 0.65, size=(T, env.action_dim))
            traj = _scripted_rollout(env, start, controls)
            label = "rollout"
        elif isinstance(env, PointMassEnv):
            if kind in (1, 2):
                goal = rng.uniform(-1.4, 1.4, size=2)
            else:
                goal = rng.choice([-1.0, 1.0], size=2) * rng.uniform(7.0, 9.0, size=2)
            traj = interpolate(start, goal, T)
            label = "line" if kind in (1, 2) else "too_fast"
        elif isinstance(env, NonholonomicCarEnv):
            if kind == 1:
                goal = np.array([rng.uniform(1.0, 2.0), rng.normal(0.0, 0.1), 0.0])
            elif kind == 2:
                goal = np.array([0.0, rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.6), 0.0])
            elif kind == 3:
                goal = np.array([rng.uniform(0.4, 0.9), rng.uniform(0.7, 1.4), 0.0])
            else:
                goal = np.array([0.4, 0.9, rng.choice([-1.0, 1.0]) * np.pi / 2])
            traj = interpolate(start, goal, T)
            label = "forward_or_sideways"
        elif isinstance(env, TwoLinkArmEnv):
            if kind in (1, 2):
                goal = env.sample_state(rng)
            elif kind == 3:
                goal = np.array([2.25, rng.uniform(-0.3, 0.6)])
            else:
                goal = start + rng.normal(0.0, 1.2, size=2)
            traj = interpolate(start, goal, T)
            label = "cartesian_line"
        elif isinstance(env, PlanarPusherEnv):
            if kind == 1:
                start = np.array([-0.24, 0.0, 0.0, 0.0])
                controls = np.tile(np.array([0.7, 0.0]), (T, 1))
                traj = _scripted_rollout(env, start, controls)
                label = "contact_rollout"
            elif kind in (2, 3):
                goal = np.array([start[0], start[1], rng.uniform(0.7, 1.2), rng.uniform(-0.5, 0.5)])
                traj = interpolate(start, goal, T)
                label = "object_without_contact"
            else:
                goal = np.array([-0.15, 0.0, 0.75, 0.0])
                traj = interpolate(start, goal, T)
                label = "move_to_push"
        else:
            raise TypeError(type(env))
        if kind == 4:
            noise = rng.normal(0.0, 0.035, size=traj.shape)
            noise[0] = 0.0
            noise[-1] = 0.0
            traj = traj + noise
        out.append({"env": env.name, "kind": label, "trajectory": traj, "goal": traj[-1]})
    return out


def score_candidates(env, n: int, T: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    example = None
    for idx, item in enumerate(candidate_trajectories(env, n, T, seed)):
        traj = item["trajectory"]
        ev = rollout_with_recovered_controls(env, traj)
        row = {
            "env": env.name,
            "candidate_id": idx,
            "kind": item["kind"],
            "failure": int(not ev["success"]),
            "success": int(ev["success"]),
            "actionability_residual": ev["actionability_residual"],
            "mean_residual": ev["mean_residual"],
            "rollout_error": ev["rollout_error"],
            "xy_error": ev["xy_error"],
            "action_norm": ev["action_norm"],
            "smoothness": smoothness_score(traj),
            "path_length": float(np.sum(np.linalg.norm(np.diff(env.plot_xy(traj), axis=0), axis=1))),
            "random_score": float(np.random.default_rng(seed + idx).random()),
        }
        row["passive_nll"] = row["smoothness"] + 0.03 * row["path_length"]
        rows.append(row)
        if example is None and row["failure"] == 1:
            example = {
                "candidate": env.plot_xy(traj),
                "rollout": env.plot_xy(ev["rollout"]),
            }
    if example is None:
        item = candidate_trajectories(env, 1, T, seed + 900)[0]
        ev = rollout_with_recovered_controls(env, item["trajectory"])
        example = {"candidate": env.plot_xy(item["trajectory"]), "rollout": env.plot_xy(ev["rollout"])}
    return pd.DataFrame(rows), example


def _repair_case(env, T: int) -> np.ndarray:
    start = env.default_start()
    if isinstance(env, PointMassEnv):
        goal = np.array([1.0, 0.75])
        traj = np.repeat(start[None, :], T + 1, axis=0)
        traj[-1] = goal
        return traj
    if isinstance(env, NonholonomicCarEnv):
        goal = np.array([0.3, 1.0, np.pi / 2])
        traj = interpolate(start, np.array([0.0, 1.0, 0.0]), T)
        traj[-1] = goal
        return traj
    if isinstance(env, TwoLinkArmEnv):
        goal = np.array([0.2, 1.55])
        traj = np.repeat(start[None, :], T + 1, axis=0)
        traj[T // 2 :] = goal
        return traj
    if isinstance(env, PlanarPusherEnv):
        goal = np.array([-0.15, 0.0, 0.65, 0.0])
        traj = interpolate(start, np.array([start[0], start[1], 0.65, 0.0]), T)
        traj[-1] = goal
        return traj
    raise TypeError(type(env))


def run_repair_suite(T: int, seed: int) -> tuple[pd.DataFrame, dict]:
    configs = {
        "goal_only": RepairConfig(alpha_goal=80.0, beta_actionability=0.0, gamma_smoothness=0.0, world_weight=0.0, maxiter=120),
        "world_goal": RepairConfig(alpha_goal=80.0, beta_actionability=0.0, gamma_smoothness=0.0, world_weight=1.0, maxiter=120),
        "world_goal_smooth": RepairConfig(alpha_goal=80.0, beta_actionability=0.0, gamma_smoothness=0.8, world_weight=1.0, maxiter=120),
        "world_goal_actionability": RepairConfig(alpha_goal=80.0, beta_actionability=24.0, gamma_smoothness=0.0, world_weight=1.0, maxiter=160),
        "full_energy": RepairConfig(alpha_goal=80.0, beta_actionability=24.0, gamma_smoothness=0.8, world_weight=1.0, maxiter=180),
    }
    rows = []
    plot_payload = {}
    for env in make_envs():
        before = _repair_case(env, T)
        before_eval = rollout_with_recovered_controls(env, before)
        rows.append(
            {
                "env": env.name,
                "variant": "candidate",
                "residual": before_eval["actionability_residual"],
                "rollout_error": before_eval["rollout_error"],
                "success": int(before_eval["success"]),
                "energy_drop": 0.0,
            }
        )
        best_after = before
        best_eval = before_eval
        for name, cfg in configs.items():
            repaired, info = repair_trajectory(env, before, before[-1], cfg)
            ev = rollout_with_recovered_controls(env, repaired)
            rows.append(
                {
                    "env": env.name,
                    "variant": name,
                    "residual": ev["actionability_residual"],
                    "rollout_error": ev["rollout_error"],
                    "success": int(ev["success"]),
                    "energy_drop": info["initial_energy"] - info["final_energy"],
                }
            )
            if name == "full_energy":
                best_after = repaired
                best_eval = ev
        plot_payload[env.name] = {
            "before": env.plot_xy(before),
            "after": env.plot_xy(best_after),
            "rollout_before": env.plot_xy(before_eval["rollout"]),
            "rollout_after": env.plot_xy(best_eval["rollout"]),
        }
    return pd.DataFrame(rows), plot_payload
