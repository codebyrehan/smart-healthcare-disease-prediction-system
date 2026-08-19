"""Smart Healthcare Data Science application entry point."""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.benchmark import load_benchmark
from src.data_pipeline import FEATURES, clean_features, data_quality_report, load_dataset
from src.data_quality import quality_summary
from src.explainability import global_feature_importance
from src.prediction import predict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
MODEL = None
DATASET = None
LOAD_ERROR = None

try:
    import joblib
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    LOAD_ERROR = str(exc)

try:
    DATASET, _ = clean_features(load_dataset())
except Exception as exc:
    LOAD_ERROR = LOAD_ERROR or str(exc)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None, "dataset_loaded": DATASET is not None})


@app.get("/api/metadata")
def metadata():
    if DATASET is None:
        return jsonify({"error": "Dataset is unavailable."}), 503
    return jsonify({"features": FEATURES, "quality": data_quality_report(DATASET), "quality_summary": quality_summary(DATASET)})


@app.get("/api/benchmark")
def benchmark():
    try:
        return jsonify(load_benchmark())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except (OSError, ValueError):
        return jsonify({"error": "Benchmark artifacts could not be read safely."}), 500


@app.get("/api/explainability")
def explainability():
    if MODEL is None:
        return jsonify({"error": "Prediction model is unavailable."}), 503
    values = global_feature_importance(MODEL, FEATURES)
    if not values:
        return jsonify({"error": "This model does not expose verified global feature importance."}), 503
    return jsonify({"model": MODEL.__class__.__name__, "importance": values})


@app.post("/api/predict")
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Prediction model is unavailable. Train and persist the verified model first."}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        result = predict(MODEL, payload)
        result["disclaimer"] = "Educational ML prediction only; not a medical diagnosis or substitute for professional care."
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
