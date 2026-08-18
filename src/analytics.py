"""Portfolio-ready analytics payloads for the Data Science dashboard."""
from __future__ import annotations

import pandas as pd

from .data_pipeline import FEATURES, TARGET
from .modeling import ModelResult


def dataset_kpis(df: pd.DataFrame) -> dict:
    outcome_rate = float(df[TARGET].mean())
    return {
        "rows": int(len(df)),
        "features": len(FEATURES),
        "positive_cases": int(df[TARGET].sum()),
        "negative_cases": int((df[TARGET] == 0).sum()),
        "positive_rate": outcome_rate,
    }


def model_leaderboard(results: dict[str, ModelResult]) -> list[dict]:
    rows = []
    for result in results.values():
        rows.append({
            "model": result.name,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "specificity": result.specificity,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
            "pr_auc": result.pr_auc,
            "cv_roc_auc_mean": result.cv_roc_auc_mean,
            "cv_roc_auc_std": result.cv_roc_auc_std,
        })
    return sorted(rows, key=lambda row: row["roc_auc"], reverse=True)


def feature_profiles(df: pd.DataFrame) -> list[dict]:
    profiles = []
    for feature in FEATURES:
        series = df[feature]
        profiles.append({
            "feature": feature,
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        })
    return profiles
