"""Reproducible model training and evaluation."""
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_pipeline import FEATURES, TARGET


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    cv_roc_auc_mean: float
    cv_roc_auc_std: float


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[dict[str, ModelResult], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), FEATURES),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}

    for name, estimator in build_models().items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(y_test, predictions, zero_division=0)),
            recall=float(recall_score(y_test, predictions, zero_division=0)),
            specificity=float(specificity),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            pr_auc=float(average_precision_score(y_test, probabilities)),
            confusion_matrix=cm.tolist(),
            cv_roc_auc_mean=float(cv_scores.mean()),
            cv_roc_auc_std=float(cv_scores.std()),
        )

    return results, X_train, y_train, X_test, y_test


def metrics_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": r.name,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "specificity": r.specificity,
            "f1": r.f1,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "cv_roc_auc_mean": r.cv_roc_auc_mean,
            "cv_roc_auc_std": r.cv_roc_auc_std,
        }
        for r in results.values()
    ]).sort_values("roc_auc", ascending=False)
