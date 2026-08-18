"""Prediction probability calibration diagnostics."""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_table(model, X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Return observed outcome frequency against predicted probability bins."""
    observed, predicted = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted_probability": predicted, "observed_frequency": observed})
