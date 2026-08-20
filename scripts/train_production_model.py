"""Train and persist the verified production prediction artifact.

This script is deterministic and uses only the repository's validated PIMA dataset.
It is executed during deployment so the service never depends on a committed binary model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_pipeline import FEATURES, TARGET, clean_features, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    raw = load_dataset(ROOT / "data" / "PIMA_Diabetes_Dataset.xlsx")
    df, quality = clean_features(raw)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    metrics = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        })
        fitted[name] = model

    selected = max(metrics, key=lambda item: item["roc_auc"])["model"]
    joblib.dump(fitted[selected], MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "model": selected,
        "model_version": "production-1",
        "selection_metric": "roc_auc",
        "random_state": 42,
        "feature_names": FEATURES,
        "dataset": "PIMA Indians Diabetes Dataset",
        "dataset_rows_after_cleaning": len(df),
        "data_quality": quality,
    }, indent=2), encoding="utf-8")
    print(f"Production model ready: {selected}")


if __name__ == "__main__":
    main()
