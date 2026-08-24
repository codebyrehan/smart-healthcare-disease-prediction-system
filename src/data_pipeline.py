"""Robust data loading, schema validation, and preprocessing pipeline for PIMA Diabetes Dataset."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
EXPECTED_COLUMNS = FEATURES + [TARGET]
IMPOSSIBLE_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def get_default_data_path() -> Path:
    """Find the dataset in standard local locations."""
    candidates = [
        Path("data/PIMA_Diabetes_Dataset.xlsx"),
        Path("data/pima_diabetes.csv"),
        Path(__file__).resolve().parents[1] / "data" / "PIMA_Diabetes_Dataset.xlsx",
        Path(__file__).resolve().parents[1] / "data" / "pima_diabetes.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the raw PIMA diabetes dataset from disk."""
    p = Path(path) if path else get_default_data_path()
    if not p.exists():
        # Fallback to CSV if xlsx is specified but not found, or vice versa
        alt = p.with_suffix(".csv") if p.suffix.lower() in {".xlsx", ".xls"} else p.with_suffix(".xlsx")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Dataset not found at {p} or {alt}.")

    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(p)
        except Exception:
            csv_path = p.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise
    else:
        df = pd.read_csv(p)

    return validate_schema(df)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required features and target exist with proper numeric types."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean_df = df[EXPECTED_COLUMNS].copy()
    for col in EXPECTED_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    if clean_df[TARGET].isna().any():
        raise ValueError("Target contains missing or non-numeric values.")

    unique_targets = set(clean_df[TARGET].dropna().unique())
    if not unique_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target contains invalid classes: {unique_targets}. Expected binary {0, 1}.")

    clean_df[TARGET] = clean_df[TARGET].astype(int)
    return clean_df.reset_index(drop=True)


def clean_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle biologically impossible zeros by replacing them with NaN and imputing with median.
    Tracks all imputation and cleaning steps for reproducible quality reporting.
    """
    validated = validate_schema(df)
    r = validated.copy()

    exact_duplicates = int(r.duplicated().sum())

    imputation_report = {}
    for col in IMPOSSIBLE_ZERO_COLUMNS:
        zero_mask = r[col] == 0
        zero_count = int(zero_mask.sum())
        r.loc[zero_mask, col] = np.nan
        median_val = float(r[col].median())
        r[col] = r[col].fillna(median_val)
        imputation_report[col] = {
            "zeros_imputed": zero_count,
            "imputed_median": round(median_val, 3),
        }

    if r[FEATURES].isna().any().any():
        raise ValueError("Unexpected missing values remain after imputation.")

    report = {
        "rows_total": len(r),
        "duplicates": exact_duplicates,
        "features_count": len(FEATURES),
        "imputation": imputation_report,
        "class_distribution": {
            "0": int((r[TARGET] == 0).sum()),
            "1": int((r[TARGET] == 1).sum()),
        },
    }
    return r, report


def data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level data quality metrics for the UI and monitoring."""
    target_series = df[TARGET] if TARGET in df.columns else None
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "class_counts": {str(k): int(v) for k, v in target_series.value_counts().sort_index().items()} if target_series is not None else {},
        "feature_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max())}
            for c in FEATURES
            if c in df.columns
        },
    }
