"""Train, evaluate, and persist the best reproducible model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from .data_pipeline import clean_features, data_quality_report, load_dataset
from .modeling import metrics_table, train_and_evaluate


def train(output_dir: str = "models") -> None:
    df = load_dataset()
    df, cleaning_report = clean_features(df)
    quality = data_quality_report(df)
    results, *_ = train_and_evaluate(df)
    metrics = metrics_table(results)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_name = str(metrics.iloc[0]["model"])
    best = results[best_name]
    joblib.dump(best.pipeline, output / "best_model.joblib")
    metrics.to_json(output / "metrics.json", orient="records", indent=2)
    metadata = {
        "model": best_name,
        "model_version": "1.0.0",
        "features": list(df.columns[:-1]),
        "target": "Outcome",
        "dataset_rows": len(df),
        "data_quality": quality,
        "cleaning": cleaning_report,
        "selection_metric": "ROC-AUC",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    train()
