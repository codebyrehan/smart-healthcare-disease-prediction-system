"""Patient parameter validation and deterministic model prediction service."""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd
from .data_pipeline import FEATURES

INPUT_RANGES: Dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 300),
    "BloodPressure": (30, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10.0, 70.0),
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age": (18, 120),
}


def validate_patient_input(values: Dict[str, Any]) -> Dict[str, float]:
    """Validate that all 8 features are present, numeric, and within clinical bounds."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing required clinical parameters: {missing}")

    validated = {}
    for f in FEATURES:
        raw_val = values.get(f)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{f}' must be a valid number.")

        lo, hi = INPUT_RANGES[f]
        if not (lo <= val <= hi):
            raise ValueError(f"'{f}' value {val} is outside clinical research range [{lo}, {hi}].")
        validated[f] = val

    return validated


def predict(model: Any, values: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run prediction on the given model with validated inputs.
    Returns canonical prediction dictionary with probability, category, and factors.
    """
    clean_inputs = validate_patient_input(values)
    df = pd.DataFrame([clean_inputs], columns=FEATURES)

    proba_array = model.predict_proba(df)
    probability = float(proba_array[0, 1])
    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk_level = "Low"
        risk_label = "Low Probability"
    elif probability < 0.60:
        risk_level = "Moderate"
        risk_label = "Moderate Probability"
    else:
        risk_level = "Elevated"
        risk_label = "Elevated Risk"

    # Identify contributing factors based on high glucose, BMI, age, etc.
    top_factors = []
    if clean_inputs["Glucose"] >= 140:
        top_factors.append(f"Elevated Plasma Glucose ({clean_inputs['Glucose']:.0f} mg/dL)")
    if clean_inputs["BMI"] >= 30.0:
        top_factors.append(f"High Body Mass Index ({clean_inputs['BMI']:.1f} kg/m²)")
    if clean_inputs["Age"] >= 45:
        top_factors.append(f"Age Category ({clean_inputs['Age']:.0f} years)")
    if clean_inputs["DiabetesPedigreeFunction"] >= 0.6:
        top_factors.append(f"Elevated Genetic Pedigree Score ({clean_inputs['DiabetesPedigreeFunction']:.2f})")
    if clean_inputs["BloodPressure"] >= 90:
        top_factors.append(f"Elevated Blood Pressure ({clean_inputs['BloodPressure']:.0f} mm Hg)")

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_percentage": round(probability * 100, 1),
        "risk_level": risk_level,
        "risk_label": risk_label,
        "threshold": threshold,
        "contributing_factors": top_factors or ["Values within baseline observed ranges"],
        "inputs": clean_inputs,
    }
