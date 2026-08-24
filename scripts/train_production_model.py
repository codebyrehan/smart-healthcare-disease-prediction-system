"""Train deterministic production models and persist canonical evaluation artifacts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure ROOT is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_pipeline import FEATURES, TARGET, clean_features, load_dataset
from src.explainability import global_feature_importance
from src.db import log_experiment

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _points(x, y, limit=80):
    if len(x) <= limit:
        return [{"x": round(float(a), 4), "y": round(float(b), 4)} for a, b in zip(x, y)]
    idx = np.linspace(0, len(x) - 1, limit).astype(int)
    return [{"x": round(float(x[i]), 4), "y": round(float(y[i]), 4)} for i in idx]


def main() -> None:
    print("==================================================")
    print("  TRAINING SMART HEALTHCARE PRODUCTION MODELS")
    print("==================================================")
    raw = load_dataset()
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
            n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    metrics_list = []
    fitted_models = {}
    evaluation_dict = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        cm = confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, probability)
        cal_true, cal_pred = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")

        acc = float(accuracy_score(y_test, prediction))
        prec = float(precision_score(y_test, prediction, zero_division=0))
        rec = float(recall_score(y_test, prediction, zero_division=0))
        f1 = float(f1_score(y_test, prediction, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, probability))
        pr_auc = float(average_precision_score(y_test, probability))

        threshold_steps = np.linspace(0.1, 0.9, 9)
        threshold_rows = []
        for t in threshold_steps:
            t_pred = (probability >= t).astype(int)
            p_val = float(precision_score(y_test, t_pred, zero_division=0))
            r_val = float(recall_score(y_test, t_pred, zero_division=0))
            f_val = float((2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0)
            threshold_rows.append({
                "threshold": round(float(t), 2),
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "f1": round(f_val, 4),
            })

        feat_imp = global_feature_importance(model, FEATURES)

        metrics_row = {
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
        }
        metrics_list.append(metrics_row)
        fitted_models[name] = model

        # Canonical structure for this model
        evaluation_dict[name] = {
            "model": name,
            "metrics": metrics_row,
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr],
                "points": _points(fpr, tpr),
            },
            "pr_curve": {
                "recall": [round(float(v), 4) for v in recall_arr],
                "precision": [round(float(v), 4) for v in precision_arr],
                "points": _points(recall_arr, precision_arr),
            },
            "calibration": {
                "predicted": [round(float(v), 4) for v in cal_pred],
                "observed": [round(float(v), 4) for v in cal_true],
                "points": _points(cal_pred, cal_true),
            },
            "thresholds": threshold_rows,
            "feature_importance": feat_imp,
        }

        # Save individual model artifact
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{slug}.joblib")

        # Log experiment to database
        try:
            log_experiment(
                model_name=name,
                metrics=metrics_row,
                hyperparameters={"random_state": 42},
                dataset_version="PIMA-Cleaned-v1.0",
                feature_count=len(FEATURES),
            )
        except Exception:
            pass

        print(f"[OK] Trained & Validated {name:<22} | Acc: {acc*100:.2f}% | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

    selected_model_name = max(metrics_list, key=lambda m: m["roc_auc"])["model"]
    best_slug = selected_model_name.lower().replace(" ", "_")
    joblib.dump(fitted_models[selected_model_name], MODEL_DIR / "best_model.joblib")

    # Persist benchmark and evaluation artifacts
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_list, indent=2), encoding="utf-8")
    (MODEL_DIR / "evaluation.json").write_text(
        json.dumps({
            "test_size": 0.2,
            "random_state": 42,
            "selected_model": selected_model_name,
            "models": evaluation_dict,
        }, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps({
            "model": selected_model_name,
            "model_version": "2026.2.0",
            "selection_metric": "roc_auc",
            "random_state": 42,
            "feature_names": FEATURES,
            "dataset": "PIMA Indians Diabetes Dataset",
            "dataset_rows_after_cleaning": len(df),
            "data_quality": quality,
        }, indent=2),
        encoding="utf-8",
    )

    print("==================================================")
    print(f"Optimal Portfolio Model Selected : {selected_model_name}")
    print("All artifacts generated in models/ directory.")
    print("==================================================")


if __name__ == "__main__":
    main()
