"""Small dependency-free experiment registry for reproducible ML runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("artifacts/experiments.json")


def record_experiment(
    *,
    model_name: str,
    metrics: dict[str, float],
    hyperparameters: dict[str, Any],
    dataset_version: str,
    feature_names: list[str],
    random_state: int = 42,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    if not model_name.strip():
        raise ValueError("model_name is required.")
    if not dataset_version.strip():
        raise ValueError("dataset_version is required.")
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")
    if not metrics:
        raise ValueError("At least one evaluation metric is required.")

    clean_metrics = {str(k): float(v) for k, v in metrics.items()}
    for name, value in clean_metrics.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Metric {name} must be between 0 and 1.")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if registry_path.exists():
        try:
            records = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Experiment registry is unreadable.") from exc
        if not isinstance(records, list):
            raise ValueError("Experiment registry must contain a JSON list.")

    experiment = {
        "experiment_id": f"exp-{len(records) + 1:04d}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_version": dataset_version,
        "feature_names": list(feature_names),
        "hyperparameters": hyperparameters,
        "metrics": clean_metrics,
        "random_state": int(random_state),
    }
    records.append(experiment)
    registry_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return experiment


def load_experiments(registry_path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Experiment registry is unreadable.") from exc
    if not isinstance(data, list):
        raise ValueError("Experiment registry must contain a JSON list.")
    return data
