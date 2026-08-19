"""Model benchmark and portfolio analytics helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_benchmark(model_dir: str = "models") -> dict:
    root = Path(model_dir)
    metrics_path = root / "metrics.json"
    metadata_path = root / "metadata.json"
    if not metrics_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Benchmark artifacts are unavailable. Run the training pipeline first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(metrics)
    if "roc_auc" in frame:
        frame = frame.sort_values("roc_auc", ascending=False)
    return {
        "selected_model": metadata.get("model"),
        "model_version": metadata.get("model_version"),
        "selection_metric": metadata.get("selection_metric"),
        "models": frame.to_dict(orient="records"),
    }
