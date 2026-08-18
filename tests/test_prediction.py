import pytest

from src.prediction import validate_patient_input


def valid_input():
    return {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.4,
        "Age": 35,
    }


def test_prediction_input_validation():
    assert validate_patient_input(valid_input())["Glucose"] == 120.0


def test_prediction_rejects_missing_feature():
    values = valid_input()
    values.pop("BMI")
    with pytest.raises(ValueError, match="Missing required"):
        validate_patient_input(values)


def test_prediction_rejects_out_of_range_feature():
    values = valid_input()
    values["Age"] = 121
    with pytest.raises(ValueError, match="between"):
        validate_patient_input(values)
