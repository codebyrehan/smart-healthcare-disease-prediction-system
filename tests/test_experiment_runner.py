from pathlib import Path

import pandas as pd

from src.experiment_runner import run_verified_experiments


def test_verified_experiments_record_all_required_models(tmp_path: Path):
    dataset = tmp_path / "pima.csv"
    rows = []
    for index in range(80):
        rows.append({
            "Pregnancies": index % 4,
            "Glucose": 90 + (index % 60),
            "BloodPressure": 60 + (index % 20),
            "SkinThickness": 20 + (index % 15),
            "Insulin": 80 + (index % 50),
            "BMI": 22 + (index % 15),
            "DiabetesPedigreeFunction": 0.2 + (index % 10) / 20,
            "Age": 21 + (index % 45),
            "Outcome": index % 2,
        })
    pd.DataFrame(rows).to_csv(dataset, index=False)
    registry = tmp_path / "experiments.json"
    records = run_verified_experiments(dataset_path=dataset, registry_path=registry)
    assert {record["model_name"] for record in records} == {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert all(0 <= record["metrics"]["roc_auc"] <= 1 for record in records)
