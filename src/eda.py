"""Robust exploratory analytics helpers for the validated dataset."""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd
from .data_pipeline import FEATURES, TARGET


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics across all 8 clinical features."""
    result = df.loc[:, FEATURES].describe().T.reset_index()
    return result.rename(columns={"index": "feature"})


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full correlation matrix including features and target outcome."""
    return df.loc[:, FEATURES + [TARGET]].corr(numeric_only=True)


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome distribution counts and percentages."""
    counts = df[TARGET].value_counts().sort_index()
    result = pd.DataFrame({
        "outcome": counts.index.astype(int),
        "count": counts.values.astype(int),
    })
    total = int(result["count"].sum())
    result["percentage"] = (result["count"] / total * 100).round(2) if total > 0 else 0.0
    return result


def feature_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Mean feature values segmented by outcome (0 vs 1)."""
    grouped = df.groupby(TARGET)[FEATURES].mean().T.reset_index()
    grouped.columns = ["feature" if c == "index" else str(c) for c in grouped.columns]
    for outcome_str in ("0", "1"):
        if outcome_str not in grouped.columns:
            grouped[outcome_str] = float("nan")
    return grouped[["feature", "0", "1"]]
