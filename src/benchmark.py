"""Model benchmark and comparative metrics helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def load_benchmark(model_dir: str | Path = "models") -> Dict[str, Any]:
    """Load benchmark metrics across all candidate models."""
    root = Path(model_dir)
    mp = root / "metrics.json"
    md = root / "metadata.json"

    if not mp.exists() or not md.exists():
        raise FileNotFoundError("Benchmark artifacts are unavailable. Run the training pipeline.")

    metrics = json.loads(mp.read_text(encoding="utf-8"))
    meta = json.loads(md.read_text(encoding="utf-8"))

    frame = pd.DataFrame(metrics)
    if "roc_auc" in frame.columns:
        frame = frame.sort_values("roc_auc", ascending=False)

    return {
        "selected_model": meta.get("model", "Random Forest"),
        "model_version": meta.get("model_version", "2026.1"),
        "selection_metric": meta.get("selection_metric", "roc_auc"),
        "dataset": meta.get("dataset", "PIMA Indians Diabetes Dataset"),
        "dataset_rows": meta.get("dataset_rows_after_cleaning", 768),
        "models": frame.to_dict(orient="records"),
    }
