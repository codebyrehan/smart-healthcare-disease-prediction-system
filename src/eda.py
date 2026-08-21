"""Robust exploratory analytics helpers for the validated dataset."""
from __future__ import annotations

import pandas as pd

from .data_pipeline import FEATURES, TARGET


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.loc[:, FEATURES].describe().T.reset_index()
    return result.rename(columns={"index": "feature"})


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, FEATURES + [TARGET]].corr(numeric_only=True)


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    result = pd.DataFrame({"outcome": counts.index.astype(int), "count": counts.values})
    result["percentage"] = result["count"] / result["count"].sum() * 100
    return result


def feature_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(TARGET)[FEATURES].mean().T.reset_index()
    grouped = grouped.rename(columns={"index": "feature"})
    for outcome in (0, 1):
        if outcome not in grouped.columns:
            grouped[outcome] = float("nan")
    return grouped[["feature", 0, 1]]
