from pathlib import Path

import pytest

from src.experiment_registry import load_experiments, record_experiment


def test_record_and_load_experiment(tmp_path: Path):
    registry = tmp_path / "experiments.json"
    result = record_experiment(
        model_name="Random Forest",
        metrics={"accuracy": 0.84, "roc_auc": 0.88},
        hyperparameters={"n_estimators": 200},
        dataset_version="pima-v1",
        feature_names=["Glucose", "BMI"],
        registry_path=registry,
    )
    records = load_experiments(registry)
    assert result["experiment_id"] == "exp-0001"
    assert records[0]["model_name"] == "Random Forest"
    assert records[0]["metrics"]["roc_auc"] == 0.88


def test_record_rejects_invalid_metric(tmp_path: Path):
    with pytest.raises(ValueError):
        record_experiment(
            model_name="Random Forest",
            metrics={"roc_auc": 1.2},
            hyperparameters={},
            dataset_version="pima-v1",
            feature_names=["Glucose"],
            registry_path=tmp_path / "experiments.json",
        )
