"""Data quality summary metrics for dataset governance."""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd
from .data_pipeline import FEATURES, TARGET


def quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce structured data quality report."""
    numeric = df[FEATURES].select_dtypes(include="number")
    finite_count = int(numeric.replace([float("inf"), float("-inf")], float("nan")).notna().sum().sum())
    total_cells = int(numeric.size)

    missing_series = df[FEATURES].isna().sum()

    return {
        "rows": int(len(df)),
        "features": len(FEATURES),
        "missing_values": int(missing_series.sum()),
        "missing_by_feature": {k: int(v) for k, v in missing_series.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "finite_value_ratio": round(float(finite_count / total_cells), 4) if total_cells else 1.0,
    }
