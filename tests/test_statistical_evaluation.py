import numpy as np
import pytest

from src.statistical_evaluation import bootstrap_roc_auc_ci


def test_bootstrap_auc_ci_is_reproducible():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_score = np.array([0.05, 0.2, 0.3, 0.7, 0.8, 0.95, 0.4, 0.6])
    first = bootstrap_roc_auc_ci(y_true, y_score, n_bootstrap=300)
    second = bootstrap_roc_auc_ci(y_true, y_score, n_bootstrap=300)
    assert first == second
    assert first["lower"] <= first["roc_auc"] <= first["upper"]


def test_bootstrap_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        bootstrap_roc_auc_ci([0, 1], [0.1, 0.9], confidence=1.0)


def test_bootstrap_rejects_too_few_iterations():
    with pytest.raises(ValueError):
        bootstrap_roc_auc_ci([0, 1], [0.1, 0.9], n_bootstrap=50)
