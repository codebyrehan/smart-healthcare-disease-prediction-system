import pandas as pd
import pytest

from src.data_pipeline import EXPECTED_COLUMNS, clean_features, validate_schema


def valid_frame():
    return pd.DataFrame({c: [1, 2] for c in EXPECTED_COLUMNS})


def test_schema_rejects_missing_column():
    df = valid_frame().drop(columns=["Glucose"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df)


def test_schema_rejects_invalid_target():
    df = valid_frame()
    df["Outcome"] = [0, 2]
    with pytest.raises(ValueError, match="only 0"):
        validate_schema(df)


def test_zero_measurements_are_imputed():
    df = valid_frame()
    df["Glucose"] = [0, 120]
    df["BMI"] = [0, 30]
    cleaned, report = clean_features(validate_schema(df))
    assert cleaned["Glucose"].isna().sum() == 0
    assert cleaned["BMI"].isna().sum() == 0
    assert report["imputed"]["Glucose"]["count"] == 1
