import pandas as pd

from src.data_pipeline import FEATURES, TARGET
from src.eda import correlation_matrix, outcome_summary, summary_statistics


def sample_df():
    data = {feature: [1.0, 2.0, 3.0, 4.0] for feature in FEATURES}
    data[TARGET] = [0, 1, 0, 1]
    return pd.DataFrame(data)


def test_eda_outputs_expected_shapes():
    df = sample_df()
    assert summary_statistics(df).shape[0] == len(FEATURES)
    assert correlation_matrix(df).shape == (len(FEATURES) + 1, len(FEATURES) + 1)
    assert outcome_summary(df)["count"].sum() == 4
