"""Global and local model explainability helpers."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from .data_pipeline import FEATURES


def global_feature_importance(model: Any, feature_names: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Extract normalized global feature importance across Logistic Regression,
    Decision Tree, and Random Forest models.
    """
    feats = feature_names or FEATURES
    if list(feats) != FEATURES:
        raise ValueError("feature_names must match project schema.")

    # Extract base estimator if wrapped in a Pipeline
    est = model
    scaler = None
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)
        scaler = model.named_steps.get("scaler", None)

    values = getattr(est, "feature_importances_", None)
    if values is None:
        coef = getattr(est, "coef_", None)
        if coef is not None:
            raw_coef = np.abs(np.asarray(coef)).ravel()
            if scaler is not None and hasattr(scaler, "scale_") and scaler.scale_ is not None:
                # Coefficients weighted by standard deviation to reflect feature impact
                values = raw_coef * np.asarray(scaler.scale_, dtype=float)
            else:
                values = raw_coef

    if values is None or len(values) != len(FEATURES):
        return []

    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    norm_values = (values / total) if total > 0 else values

    pairs = sorted(zip(FEATURES, norm_values), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]
