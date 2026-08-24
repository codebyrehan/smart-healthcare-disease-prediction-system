"""What-if and parameter sensitivity analysis engine."""
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from .data_pipeline import FEATURES


def sensitivity_analysis(
    model: Any,
    baseline: Dict[str, float],
    feature: str,
    values: List[float],
) -> List[Dict[str, float]]:
    """
    Evaluate how predicted risk shifts when a single parameter varies,
    keeping all other patient parameters held constant at baseline.
    """
    if feature not in FEATURES:
        raise ValueError(f"Feature '{feature}' is not in the project schema.")

    if not set(FEATURES).issubset(set(baseline.keys())):
        raise ValueError("Baseline must contain all 8 project features.")

    # Create batch dataframe for vector prediction
    rows = []
    for val in values:
        row = dict(baseline)
        row[feature] = float(val)
        rows.append([row[f] for f in FEATURES])

    df = pd.DataFrame(rows, columns=FEATURES)
    probas = model.predict_proba(df)[:, 1]

    results = []
    for val, p in zip(values, probas):
        results.append({
            "value": round(float(val), 2),
            "probability": round(float(p), 4),
            "risk_percentage": round(float(p) * 100, 1),
        })

    return results
