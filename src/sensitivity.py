"""Model sensitivity utilities for controlled what-if analysis."""
from __future__ import annotations

from typing import Any

from .data_pipeline import FEATURES


def sensitivity_analysis(model: Any, baseline: dict[str, float], feature: str, values: list[float]) -> list[dict[str, float]]:
    if feature not in FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")
    if set(baseline) != set(FEATURES):
        raise ValueError("Baseline must contain exactly the validated model features.")
    if not values:
        raise ValueError("At least one what-if value is required.")
    rows = []
    for value in values:
        candidate = dict(baseline)
        candidate[feature] = float(value)
        ordered = [[candidate[name] for name in FEATURES]]
        probability = float(model.predict_proba(ordered)[0][1])
        rows.append({"value": float(value), "probability": probability})
    return rows
