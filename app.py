"""Smart Healthcare Data Science application entry point."""
from __future__ import annotations

import html
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from src.benchmark import load_benchmark
from src.data_pipeline import FEATURES, TARGET, clean_features, data_quality_report, load_dataset
from src.data_quality import quality_summary
from src.eda import correlation_matrix, feature_by_outcome, outcome_summary, summary_statistics
from src.experiment_registry import load_experiments
from src.explainability import global_feature_importance
from src.prediction import predict
from src.sensitivity import sensitivity_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
MODEL = None
DATASET = None
LOAD_ERROR = None
MODEL_REGISTRY = {}
MODEL_ALIASES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}

try:
    DATASET, _ = clean_features(load_dataset())
except Exception as exc:
    LOAD_ERROR = str(exc)


def load_or_train_model():
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if DATASET is None:
        return None
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(DATASET[FEATURES], DATASET[TARGET])
    return model


def build_model_registry():
    """Build the three project models so model selection is functional at runtime."""
    if DATASET is None:
        return {}
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    }


try:
    MODEL = load_or_train_model()
    MODEL_REGISTRY = build_model_registry()
    if DATASET is not None:
        for _model in MODEL_REGISTRY.values():
            _model.fit(DATASET[FEATURES], DATASET[TARGET])
    if MODEL is not None and "random_forest" not in MODEL_REGISTRY:
        MODEL_REGISTRY["random_forest"] = MODEL
except Exception as exc:
    LOAD_ERROR = LOAD_ERROR or str(exc)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "dataset_loaded": DATASET is not None,
        "available_models": list(MODEL_ALIASES),
    })


@app.get("/api/models")
def models():
    return jsonify({"models": list(MODEL_ALIASES), "default": "Logistic Regression"})


@app.get("/api/metadata")
def metadata():
    if DATASET is None:
        return jsonify({"error": "Dataset is unavailable."}), 503
    return jsonify({"features": FEATURES, "quality": data_quality_report(DATASET), "quality_summary": quality_summary(DATASET)})


@app.get("/api/eda")
def eda():
    if DATASET is None:
        return jsonify({"error": "Validated dataset is unavailable."}), 503
    try:
        stats = summary_statistics(DATASET).round(4).to_dict(orient="records")
        corr = correlation_matrix(DATASET).round(4).fillna(0).to_dict()
        outcomes = outcome_summary(DATASET).round(2).to_dict(orient="records")
        grouped = feature_by_outcome(DATASET).round(4).to_dict(orient="records")
        return jsonify({"rows": len(DATASET), "statistics": stats, "correlation": corr, "outcomes": outcomes, "feature_by_outcome": grouped})
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "EDA analytics could not be computed safely."}), 500


@app.get("/api/report")
def report():
    if DATASET is None:
        return jsonify({"error": "Validated dataset is unavailable."}), 503
    stats = summary_statistics(DATASET).round(3).to_dict(orient="records")
    outcomes = outcome_summary(DATASET).round(2).to_dict(orient="records")
    rows = "".join(f"<tr><td>{html.escape(str(r['feature']))}</td><td>{r['mean']:.3f}</td><td>{r['std']:.3f}</td><td>{r['min']:.3f}</td><td>{r['max']:.3f}</td></tr>" for r in stats)
    classes = "".join(f"<li>Outcome {r['outcome']}: {r['count']} records ({r['percentage']:.2f}%)</li>" for r in outcomes)
    body = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Smart Healthcare Analysis Report</title><style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}h1{{margin-bottom:4px}}.note{{padding:12px;background:#f4f6f8}}</style></head><body><h1>Smart Healthcare — Data Science Report</h1><p>Validated PIMA Indians Diabetes Dataset · {len(DATASET)} records · {len(FEATURES)} predictive features</p><h2>Outcome distribution</h2><ul>{classes}</ul><h2>Descriptive statistics</h2><table><thead><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead><tbody>{rows}</tbody></table><p class='note'><strong>Responsible use:</strong> This is an educational machine-learning analysis. Model predictions do not constitute a medical diagnosis or treatment recommendation.</p></body></html>"""
    return Response(body, mimetype="text/html", headers={"Content-Disposition": "attachment; filename=smart-healthcare-report.html"})


@app.get("/api/benchmark")
def benchmark():
    try:
        return jsonify(load_benchmark())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except (OSError, ValueError):
        return jsonify({"error": "Benchmark artifacts could not be read safely."}), 500


@app.get("/api/experiments")
def experiments():
    try:
        records = load_experiments()
        public = [{"experiment_id": item.get("experiment_id"), "timestamp_utc": item.get("timestamp_utc"), "model_name": item.get("model_name"), "dataset_version": item.get("dataset_version"), "feature_count": len(item.get("feature_names", [])), "metrics": item.get("metrics", {}), "random_state": item.get("random_state")} for item in records]
        return jsonify({"experiments": public})
    except ValueError:
        return jsonify({"error": "Experiment registry is unavailable or invalid."}), 503


@app.get("/api/explainability")
def explainability():
    if MODEL is None:
        return jsonify({"error": "Prediction model is unavailable."}), 503
    values = global_feature_importance(MODEL, FEATURES)
    if not values:
        return jsonify({"error": "This model does not expose verified global feature importance."}), 503
    return jsonify({"model": MODEL.__class__.__name__, "importance": values})


@app.post("/api/sensitivity")
def sensitivity():
    if not MODEL_REGISTRY:
        return jsonify({"error": "Prediction models are unavailable."}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    feature, baseline, values = payload.get("feature"), payload.get("baseline"), payload.get("values")
    model_name = payload.get("model", "Logistic Regression")
    key = MODEL_ALIASES.get(model_name)
    if key not in MODEL_REGISTRY:
        return jsonify({"error": "Unsupported model selection."}), 400
    if not isinstance(feature, str) or not isinstance(baseline, dict) or not isinstance(values, list):
        return jsonify({"error": "feature, baseline, and values are required."}), 400
    if len(values) > 25:
        return jsonify({"error": "A maximum of 25 what-if values is allowed."}), 400
    try:
        clean_baseline = {name: float(baseline[name]) for name in FEATURES}
        clean_values = [float(value) for value in values]
        result = sensitivity_analysis(MODEL_REGISTRY[key], clean_baseline, feature, clean_values)
        return jsonify({"feature": feature, "model": model_name, "results": result, "disclaimer": "Sensitivity analysis describes model behavior; it does not establish causation or provide medical advice."})
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/predict")
def api_predict():
    if not MODEL_REGISTRY:
        return jsonify({"error": "Prediction models are unavailable."}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    model_name = payload.pop("model", "Logistic Regression")
    key = MODEL_ALIASES.get(model_name)
    if key not in MODEL_REGISTRY:
        return jsonify({"error": "Unsupported model selection."}), 400
    try:
        result = predict(MODEL_REGISTRY[key], payload)
        result.update({"model": model_name, "disclaimer": "Educational ML prediction only; not a medical diagnosis or substitute for professional care."})
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)