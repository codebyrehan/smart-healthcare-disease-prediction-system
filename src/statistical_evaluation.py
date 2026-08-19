"""Statistical evaluation helpers for reproducible model assessment."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_roc_auc_ci(
    y_true: Any,
    y_score: Any,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estimate a percentile bootstrap CI for ROC-AUC.

    Degenerate resamples (containing one class) are skipped because ROC-AUC
    is undefined for them. The function never substitutes a fabricated value.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0] or y_true.shape[0] < 2:
        raise ValueError("y_true and y_score must have the same length and contain at least two rows.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for a useful interval.")

    observed = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        if np.unique(sample_y).size < 2:
            continue
        scores.append(float(roc_auc_score(sample_y, y_score[indices])))

    if len(scores) < max(100, n_bootstrap // 4):
        raise ValueError("Too few valid bootstrap samples to estimate a reliable confidence interval.")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {
        "roc_auc": observed,
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_samples": len(scores),
        "random_state": random_state,
    }
