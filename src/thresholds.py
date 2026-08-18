"""Threshold analysis for transparent classification trade-offs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def threshold_analysis(probabilities, y_true, thresholds=None) -> pd.DataFrame:
    """Compare sensitivity/specificity across classification thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append({
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "positive_rate": float(predictions.mean()),
        })
    return pd.DataFrame(rows)
