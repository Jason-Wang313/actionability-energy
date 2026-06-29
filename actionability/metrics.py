from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def calibration_bins(y_true: np.ndarray, score: np.ndarray, bins: int = 8) -> list[dict]:
    y_true = np.asarray(y_true).astype(float)
    score = np.asarray(score, dtype=float)
    if np.allclose(score.max(), score.min()):
        normalized = np.zeros_like(score)
    else:
        normalized = (score - score.min()) / (score.max() - score.min())
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (normalized >= lo) & (normalized <= hi if hi == 1.0 else normalized < hi)
        if not np.any(mask):
            continue
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_score": float(normalized[mask].mean()),
                "failure_rate": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return out


def threshold_accuracy(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    if len(score) == 0:
        return float("nan")
    thresh = float(np.median(score))
    pred = (score >= thresh).astype(int)
    return float((pred == y_true).mean())
