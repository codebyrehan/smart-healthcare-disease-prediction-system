"""Reproducible experiment runner using the project's verified data pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_pipeline import FEATURES, TARGET, clean_features, load_dataset
from .experiment_registry import record_experiment

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def run_verified_experiments(*, dataset_path: str | Path = "data/PIMA_Diabetes_Dataset.xlsx", dataset_version: str = "pima-v1", registry_path: Path = Path("artifacts/experiments.json")) -> list[dict[str, Any]]:
    """Train the three in-scope models and persist only measured evaluation results."""
    df, _ = clean_features(load_dataset(dataset_path))
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    records = []
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        predicted = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
        }
        record = record_experiment(
            model_name=name,
            metrics=metrics,
            hyperparameters=model.get_params(deep=False),
            dataset_version=dataset_version,
            feature_names=FEATURES,
            random_state=42,
            registry_path=registry_path,
        )
        records.append(record)
    return records
