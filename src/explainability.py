"""Model-agnostic and tree-based feature attribution helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from .data_pipeline import FEATURES


def permutation_importance_table(model, X: pd.DataFrame, y: pd.Series, repeats: int = 20) -> pd.DataFrame:
    """Estimate feature importance without depending on model internals."""
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
