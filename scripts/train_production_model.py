"""Train deterministic production models and persist evaluation artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.data_pipeline import FEATURES, TARGET, clean_features, load_dataset

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _points(x, y, limit=80):
    if len(x) <= limit:
        return [{"x": float(a), "y": float(b)} for a, b in zip(x, y)]
    idx = np.linspace(0, len(x) - 1, limit).astype(int)
    return [{"x": float(x[i]), "y": float(y[i])} for i in idx]


def main() -> None:
    raw = load_dataset(ROOT / "data" / "PIMA_Diabetes_Dataset.xlsx")
    df, quality = clean_features(raw)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    metrics, fitted, evaluation = [], {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        cm = confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        cal_true, cal_pred = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
        row = {"model": name, "accuracy": float(accuracy_score(y_test, prediction)), "precision": float(precision_score(y_test, prediction, zero_division=0)), "recall": float(recall_score(y_test, prediction, zero_division=0)), "f1": float(f1_score(y_test, prediction, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, probability)), "pr_auc": float(average_precision_score(y_test, probability))}
        metrics.append(row)
        fitted[name] = model
        thresholds = []
        for threshold in np.linspace(0.1, 0.9, 9):
            pred_t = (probability >= threshold).astype(int)
            p = precision_score(y_test, pred_t, zero_division=0)
            r = recall_score(y_test, pred_t, zero_division=0)
            thresholds.append({"threshold": float(threshold), "precision": float(p), "recall": float(r), "f1": float((2*p*r/(p+r)) if p+r else 0)})
        evaluation[name] = {"confusion_matrix": cm, "roc_curve": _points(fpr, tpr), "pr_curve": _points(recall, precision), "calibration": _points(cal_pred, cal_true), "thresholds": thresholds}
    selected = max(metrics, key=lambda item: item["roc_auc"])["model"]
    joblib.dump(fitted[selected], MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps({"test_size": 0.2, "random_state": 42, "selected_model": selected, "models": evaluation}, indent=2), encoding="utf-8")
    (MODEL_DIR / "metadata.json").write_text(json.dumps({"model": selected, "model_version": "production-2", "selection_metric": "roc_auc", "random_state": 42, "feature_names": FEATURES, "dataset": "PIMA Indians Diabetes Dataset", "dataset_rows_after_cleaning": len(df), "data_quality": quality}, indent=2), encoding="utf-8")
    print(f"Production model ready: {selected}")


if __name__ == "__main__":
    main()
