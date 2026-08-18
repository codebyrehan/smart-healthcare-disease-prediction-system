"""Exploratory data analysis utilities for the PIMA dataset."""
from __future__ import annotations

import pandas as pd

from .data_pipeline import FEATURES, TARGET


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics with explicit numeric coverage."""
    return df[FEATURES].describe().T.reset_index(names="feature")


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature/target Pearson correlations."""
    return df[FEATURES + [TARGET]].corr(numeric_only=True)


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return outcome counts and percentages."""
    counts = df[TARGET].value_counts().sort_index()
    result = pd.DataFrame({"outcome": counts.index.astype(int), "count": counts.values})
    result["percentage"] = result["count"] / result["count"].sum() * 100
    return result


def feature_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature means grouped by diabetes outcome."""
    return df.groupby(TARGET)[FEATURES].mean().T.reset_index(names="feature")
