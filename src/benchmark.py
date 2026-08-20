"""Model benchmark helpers."""
from pathlib import Path
import json
import pandas as pd
def load_benchmark(model_dir="models"):
    root=Path(model_dir); mp=root/"metrics.json"; md=root/"metadata.json"
    if not mp.exists() or not md.exists(): raise FileNotFoundError("Benchmark artifacts are unavailable.")
    metrics=json.loads(mp.read_text()); meta=json.loads(md.read_text()); frame=pd.DataFrame(metrics)
    if "roc_auc" in frame: frame=frame.sort_values("roc_auc",ascending=False)
    return {"selected_model":meta.get("model"),"model_version":meta.get("model_version"),"selection_metric":meta.get("selection_metric"),"models":frame.to_dict(orient="records")}
