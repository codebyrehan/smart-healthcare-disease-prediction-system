"""Controlled hyperparameter tuning for the required ML models."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_pipeline import FEATURES, TARGET


def tune_models(df: pd.DataFrame) -> dict[str, GridSearchCV]:
    """Tune the three required models using ROC-AUC and stratified CV."""
    X, y = df[FEATURES], df[TARGET].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 3, 5]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
        ),
    }
    searches = {}
    for name, (estimator, params) in models.items():
        search = GridSearchCV(estimator, params, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(X, y)
        searches[name] = search
    return searches
