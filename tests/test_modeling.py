import pandas as pd

from src.data_pipeline import FEATURES, TARGET
from src.modeling import train_and_evaluate


def test_all_required_models_train():
    rows = []
    for i in range(80):
        rows.append({
            "Pregnancies": i % 6,
            "Glucose": 90 + (i % 50),
            "BloodPressure": 60 + (i % 25),
            "SkinThickness": 15 + (i % 20),
            "Insulin": 60 + (i % 100),
            "BMI": 22 + (i % 18) * 0.5,
            "DiabetesPedigreeFunction": 0.2 + (i % 10) * 0.05,
            "Age": 21 + (i % 45),
            "Outcome": i % 2,
        })
    df = pd.DataFrame(rows)
    results, *_ = train_and_evaluate(df)
    assert set(results) == {"Logistic Regression", "Decision Tree", "Random Forest"}
    for result in results.values():
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.roc_auc <= 1
        assert result.confusion_matrix
