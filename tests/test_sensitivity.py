import pytest

from src.sensitivity import sensitivity_analysis


class StubModel:
    def predict_proba(self, rows):
        value = rows[0][1]
        probability = max(0.0, min(1.0, value / 200.0))
        return [[1.0 - probability, probability]]


def test_sensitivity_requires_known_feature():
    with pytest.raises(ValueError):
        sensitivity_analysis(StubModel(), {"Pregnancies": 1, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25, "DiabetesPedigreeFunction": 0.4, "Age": 30}, "Unknown", [90])


def test_sensitivity_returns_probability_for_each_value():
    baseline = {"Pregnancies": 1, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25, "DiabetesPedigreeFunction": 0.4, "Age": 30}
    rows = sensitivity_analysis(StubModel(), baseline, "Glucose", [90, 120, 150])
    assert [row["value"] for row in rows] == [90.0, 120.0, 150.0]
    assert all(0.0 <= row["probability"] <= 1.0 for row in rows)
