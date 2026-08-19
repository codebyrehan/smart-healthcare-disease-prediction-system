"""Model-agnostic and tree-based feature attribution helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data_pipeline import FEATURES


def permutation_importance_table(model, X: pd.DataFrame, y: pd.Series, repeats: int = 20) -> pd.DataFrame:
    """Estimate feature importance without depending on model internals."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=repeats, random_state=42, n_jobs=-1
    )
    return (
        pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(model) -> pd.DataFrame:
    """Return native tree importance when the final estimator supports it."""
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError("The supplied estimator does not expose tree feature importance.")
    return (
        pd.DataFrame({"feature": FEATURES, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def model_native_importance(model) -> list[dict[str, float | str]]:
    """Return normalized native importances when supported; otherwise return []."""
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is not None:
            values = np.abs(np.asarray(coefficients)).ravel()
    if values is None:
        return []
    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(FEATURES):
        return []
    total = values.sum()
    normalized = values / total if total > 0 else values
    return [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(FEATURES, normalized), key=lambda item: item[1], reverse=True)
    ]
