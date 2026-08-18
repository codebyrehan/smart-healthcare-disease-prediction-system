"""Safe prediction service around a trained model."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .data_pipeline import FEATURES

INPUT_RANGES = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 400),
    "BloodPressure": (1, 250),
    "SkinThickness": (0, 150),
    "Insulin": (0, 1000),
    "BMI": (1, 100),
    "DiabetesPedigreeFunction": (0, 5),
    "Age": (1, 120),
}


def validate_patient_input(values: dict[str, Any]) -> dict[str, float]:
    """Validate and normalize one prediction request without accepting extras."""
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    normalized: dict[str, float] = {}
    for feature in FEATURES:
        try:
            value = float(values[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc
        if not value == value:  # NaN
            raise ValueError(f"{feature} cannot be NaN.")
        low, high = INPUT_RANGES[feature]
        if not low <= value <= high:
            raise ValueError(f"{feature} must be between {low} and {high}.")
        normalized[feature] = value
    return normalized


def predict(model, values: dict[str, Any]) -> dict[str, Any]:
    """Generate a prediction and probability from an already-trained pipeline."""
    validated = validate_patient_input(values)
    frame = pd.DataFrame([validated], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Higher likelihood" if prediction else "Lower likelihood",
        "probability": probability,
        "threshold": 0.5,
    }
