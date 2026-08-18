"""Data loading and validation for the PIMA diabetes dataset."""
from pathlib import Path
import pandas as pd
import numpy as np

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"
EXPECTED_COLUMNS = FEATURES + [TARGET]
IMPOSSIBLE_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: str | Path = "data/PIMA_Diabetes_Dataset.xlsx") -> pd.DataFrame:
    """Load the repository dataset; never silently fabricate patient data."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Provide the verified PIMA dataset before training."
        )
    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or Excel.")
    return validate_schema(df)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, target encoding, and numeric feature types."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    result = df[EXPECTED_COLUMNS].copy()
    for column in EXPECTED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[TARGET].isna().any():
        raise ValueError("Target contains missing or non-numeric values.")
    invalid_target = ~result[TARGET].isin([0, 1])
    if invalid_target.any():
        raise ValueError("Outcome must contain only 0 (non-diabetic) and 1 (diabetic).")
    if result.duplicated().any():
        result = result.drop_duplicates().reset_index(drop=True)
    return result


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace medically impossible zero measurements with NaN and median-impute."""
    result = df.copy()
    report = {"rows_before": len(result), "duplicates_removed": 0, "imputed": {}}
    before = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(result)

    for column in IMPOSSIBLE_ZERO_COLUMNS:
        zero_count = int((result[column] == 0).sum())
        result.loc[result[column] == 0, column] = np.nan
        median = float(result[column].median())
        result[column] = result[column].fillna(median)
        report["imputed"][column] = {"count": zero_count, "median": median}

    if result[FEATURES].isna().any().any():
        raise ValueError("Unexpected missing feature values remain after preprocessing.")
    report["rows_after"] = len(result)
    return result, report


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return reproducible dataset-quality metadata for analysis and documentation."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "feature_ranges": {
            column: {"min": float(df[column].min()), "max": float(df[column].max())}
            for column in FEATURES
        },
    }
