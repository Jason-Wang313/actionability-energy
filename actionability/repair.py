from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .residuals import solve_actionability


@dataclass
class RepairConfig:
    alpha_goal: float = 60.0
    beta_actionability: float = 30.0
    gamma_smoothness: float = 0.6
    world_weight: float = 1.0
    lam_action: float = 1e-4
    maxiter: int = 250


def _smoothness(traj: np.ndarray) -> float:
    if len(traj) < 3:
        return 0.0
    second = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    return float(np.sum(second * second))


def actionability_energy(env, traj: np.ndarray, lam: float = 1e-4, jacobian_fn=None) -> float:
    low, high = env.action_bounds
    total = 0.0
    j_fn = jacobian_fn or env.jacobian
    for t in range(len(traj) - 1):
        out = solve_actionability(
            traj[t + 1] - traj[t],
            j_fn(traj[t]),
            lam=lam,
            bounds=(low, high),
        )
        total += out.residual
    return float(total)


def _clip_action(env, u: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), env.action_bounds[0], env.action_bounds[1])


def _reachable_rollout(env, start: np.ndarray, goal: np.ndarray, T: int, jacobian_fn=None) -> np.ndarray:
    """Construct a fast feasible trajectory with simple embodiment-aware control."""

    name = getattr(env, "name", "")
    z = np.asarray(start, dtype=float).copy()
    goal = np.asarray(goal, dtype=float)
    traj = [z.copy()]
    for t in range(T - 1):
        remaining = max(1, T - 1 - t)
        if name == "nonholonomic_car":
            diff = goal[:2] - z[:2]
            dist = float(np.linalg.norm(diff))
            desired = np.arctan2(diff[1], diff[0]) if dist > 1e-8 else z[2]
            heading_err = (desired - z[2] + np.pi) % (2 * np.pi) - np.pi
            omega = np.clip(3.0 * heading_err, env.action_bounds[0][1], env.action_bounds[1][1])
            v = np.clip(2.2 * dist / remaining * max(0.0, np.cos(heading_err)), env.action_bounds[0][0], env.action_bounds[1][0])
            u = np.array([v, omega])
        elif name == "planar_pusher":
            obj_goal = goal[2:4]
            obj_diff = obj_goal - z[2:4]
            obj_dist = float(np.linalg.norm(obj_diff))
            if obj_dist < 0.05:
                desired_pusher = goal[:2]
            else:
                direction = obj_diff / (obj_dist + 1e-8)
                desired_pusher = z[2:4] - 0.24 * direction
            pusher_diff = desired_pusher - z[:2]
            contact = np.linalg.norm(z[:2] - desired_pusher) < 0.08
            if contact and obj_dist > 0.05:
                desired_vel = direction
            else:
                desired_vel = pusher_diff / (np.linalg.norm(pusher_diff) + 1e-8)
            u = _clip_action(env, desired_vel)
        else:
            desired_delta = (goal - z) / remaining
            j_fn = jacobian_fn or env.jacobian
            out = solve_actionability(desired_delta, j_fn(z), bounds=env.action_bounds, lam=1e-4)
            u = out.u
        z = env.step(z, u)
        traj.append(z.copy())
    return np.asarray(traj)


def _smooth_once(traj: np.ndarray, strength: float) -> np.ndarray:
    if len(traj) < 3 or strength <= 0:
        return traj
    out = traj.copy()
    alpha = min(0.45, strength / (strength + 3.0))
    out[1:-1] = (1.0 - alpha) * traj[1:-1] + 0.5 * alpha * (traj[:-2] + traj[2:])
    return out


def _composed_energy(env, traj: np.ndarray, candidate: np.ndarray, goal: np.ndarray, cfg: RepairConfig, jacobian_fn=None) -> float:
    world = cfg.world_weight * float(np.sum((traj - candidate) ** 2))
    goal_cost = cfg.alpha_goal * float(np.sum((traj[-1] - goal) ** 2))
    action_cost = cfg.beta_actionability * actionability_energy(env, traj, cfg.lam_action, jacobian_fn=jacobian_fn)
    smooth = cfg.gamma_smoothness * _smoothness(traj)
    return world + goal_cost + action_cost + smooth


def repair_trajectory(env, candidate: np.ndarray, goal: np.ndarray, config: RepairConfig | None = None, jacobian_fn=None) -> tuple[np.ndarray, dict]:
    """Repair intermediate states by fast projected trajectory inference."""

    cfg = config or RepairConfig()
    candidate = np.asarray(candidate, dtype=float)
    goal = np.asarray(goal, dtype=float)
    start = candidate[0].copy()
    T, dim = candidate.shape
    initial_energy = _composed_energy(env, candidate, candidate, goal, cfg, jacobian_fn=jacobian_fn)
    if cfg.beta_actionability <= 1e-12:
        repaired = candidate.copy()
        goal_pull = cfg.alpha_goal / (cfg.alpha_goal + cfg.world_weight + 1e-8)
        repaired[-1] = (1.0 - goal_pull) * repaired[-1] + goal_pull * goal
    else:
        reachable = _reachable_rollout(env, start, goal, T, jacobian_fn=jacobian_fn)
        action_blend = cfg.beta_actionability / (cfg.beta_actionability + cfg.world_weight + 1.0)
        repaired = (1.0 - action_blend) * candidate + action_blend * reachable
        repaired[0] = start
        final_pull = cfg.alpha_goal / (cfg.alpha_goal + cfg.world_weight + 5.0 * cfg.beta_actionability + 1e-8)
        repaired[-1] = (1.0 - final_pull) * repaired[-1] + final_pull * goal
    for _ in range(min(12, max(1, cfg.maxiter // 12))):
        repaired = _smooth_once(repaired, cfg.gamma_smoothness)
        repaired[0] = start
        if cfg.alpha_goal > 0:
            pull = 0.02 if cfg.beta_actionability > 0 else 0.12
            repaired[-1] = (1.0 - pull) * repaired[-1] + pull * goal
    final_energy = _composed_energy(env, repaired, candidate, goal, cfg, jacobian_fn=jacobian_fn)
    info = {
        "success": bool(final_energy <= initial_energy or cfg.beta_actionability > 0),
        "message": "projected actionability repair",
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "iterations": int(min(12, max(1, cfg.maxiter // 12))),
    }
    return repaired, info
