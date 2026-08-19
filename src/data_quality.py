"""Dataset quality metrics for the analytics dashboard."""
from __future__ import annotations

import pandas as pd

from .data_pipeline import FEATURES


def quality_summary(df: pd.DataFrame) -> dict:
    """Return deterministic, descriptive quality checks without altering the data."""
    missing = df[FEATURES].isna().sum()
    duplicates = int(df.duplicated().sum())
    numeric = df[FEATURES].select_dtypes(include="number")
    finite_values = int(numeric.replace([float("inf"), float("-inf")], pd.NA).notna().sum().sum())
    numeric_values = int(numeric.size)
    return {
        "rows": int(len(df)),
        "features": len(FEATURES),
        "missing_values": int(missing.sum()),
        "missing_by_feature": {k: int(v) for k, v in missing.items()},
        "duplicate_rows": duplicates,
        "finite_value_ratio": float(finite_values / numeric_values) if numeric_values else 1.0,
    }
