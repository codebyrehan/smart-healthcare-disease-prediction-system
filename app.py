"""Smart Healthcare Clinical Intelligence & Disease Prediction Application."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
from flask import Flask, Response, jsonify, render_template, request

from src.benchmark import load_benchmark
from src.data_pipeline import FEATURES, TARGET, clean_features, data_quality_report, load_dataset
from src.data_quality import quality_summary
from src.db import get_recent_predictions, is_postgres, log_prediction
from src.eda import correlation_matrix, feature_by_outcome, outcome_summary, summary_statistics
from src.explainability import global_feature_importance
from src.prediction import predict, validate_patient_input
from src.reporting import generate_pdf_report
from src.sensitivity import sensitivity_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"

# Available Models Registry
MODEL_REGISTRY: Dict[str, Any] = {}
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
DEFAULT_MODEL = "Random Forest"

DATASET = None
DATASET_QUALITY = {}
LOAD_ERROR = None

# Initialize Dataset and Models
try:
    raw_df = load_dataset()
    DATASET, DATASET_QUALITY = clean_features(raw_df)
except Exception as exc:
    LOAD_ERROR = f"Dataset loading error: {exc}"


def load_model_artifacts():
    """Load serialized model artifacts or train them on-the-fly if absent."""
    global MODEL_REGISTRY
    slug_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
    }

    all_exist = all((MODEL_DIR / filename).exists() for filename in slug_map.values())
    if all_exist:
        for name, filename in slug_map.items():
            MODEL_REGISTRY[name] = joblib.load(MODEL_DIR / filename)
    else:
        # Train and serialize on-the-fly
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        if DATASET is not None:
            X, y = DATASET[FEATURES], DATASET[TARGET].astype(int)
            lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))])
            lr.fit(X, y)
            dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
            dt.fit(X, y)
            rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(X, y)

            MODEL_REGISTRY["Logistic Regression"] = lr
            MODEL_REGISTRY["Decision Tree"] = dt
            MODEL_REGISTRY["Random Forest"] = rf


try:
    load_model_artifacts()
except Exception as exc:
    LOAD_ERROR = f"{LOAD_ERROR or ''} Model loading error: {exc}".strip()


@app.after_request
def add_cache_headers(response):
    """Ensure API responses are never cached by intermediate proxies or browsers."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
def health():
    """System and infrastructure health endpoint."""
    return jsonify({
        "status": "ok",
        "models_loaded": len(MODEL_REGISTRY) == 3,
        "available_models": list(MODEL_REGISTRY.keys()),
        "default_model": DEFAULT_MODEL,
        "dataset_loaded": DATASET is not None,
        "database_type": "PostgreSQL" if is_postgres() else "SQLite",
        "error": LOAD_ERROR,
    })


@app.get("/api/models")
def list_models():
    """List available classification models and active default."""
    return jsonify({
        "models": list(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else MODEL_NAMES,
        "default": DEFAULT_MODEL if DEFAULT_MODEL in MODEL_REGISTRY else "Logistic Regression",
    })


@app.get("/api/metadata")
def metadata():
    """Data quality and schema definition."""
    if DATASET is None:
        return jsonify({"error": "Dataset is unavailable.", "details": LOAD_ERROR}), 503
    return jsonify({
        "features": FEATURES,
        "target": TARGET,
        "quality": data_quality_report(DATASET),
        "quality_summary": quality_summary(DATASET),
    })


@app.get("/api/eda")
def eda():
    """Comprehensive exploratory data analytics for validated dataset."""
    if DATASET is None:
        return jsonify({"error": "Dataset is unavailable."}), 503
    try:
        stats = summary_statistics(DATASET).round(4).to_dict(orient="records")
        corr = correlation_matrix(DATASET).round(4).fillna(0).to_dict()
        outcomes = outcome_summary(DATASET).round(2).to_dict(orient="records")
        grouped = feature_by_outcome(DATASET).round(4).to_dict(orient="records")
        return jsonify({
            "rows": len(DATASET),
            "statistics": stats,
            "correlation": corr,
            "outcomes": outcomes,
            "feature_by_outcome": grouped,
        })
    except Exception as exc:
        return jsonify({"error": f"EDA analytics could not be computed: {exc}"}), 500


@app.get("/api/evaluation")
def evaluation():
    """Canonical model evaluation evidence on fixed stratified test split."""
    eval_path = MODEL_DIR / "evaluation.json"
    if not eval_path.exists():
        return jsonify({"error": "Evaluation artifacts are unavailable. Run train_production_model.py."}), 503
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        requested_model = request.args.get("model")
        if requested_model and requested_model in payload.get("models", {}):
            payload["selected_model"] = requested_model
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Evaluation artifact invalid: {exc}"}), 500


@app.get("/api/benchmark")
def benchmark():
    """Comparative performance benchmark matrix across candidate models."""
    try:
        return jsonify(load_benchmark(MODEL_DIR))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/explainability")
def explainability():
    """Global feature importance for requested or active model."""
    requested_model = request.args.get("model", DEFAULT_MODEL)
    if requested_model not in MODEL_REGISTRY:
        requested_model = next(iter(MODEL_REGISTRY.keys()), DEFAULT_MODEL)

    model = MODEL_REGISTRY.get(requested_model)
    if model is None:
        return jsonify({"error": f"Model '{requested_model}' is unavailable."}), 503

    importance = global_feature_importance(model, FEATURES)
    return jsonify({
        "model": requested_model,
        "importance": importance,
    })


@app.post("/api/predict")
def api_predict():
    """Run diabetes risk prediction on validated clinical inputs with selected model."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    model_name = payload.pop("model", DEFAULT_MODEL)
    if model_name not in MODEL_REGISTRY:
        # Fallback to available model if exact match not found
        model_name = next(iter(MODEL_REGISTRY.keys()), DEFAULT_MODEL)

    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        return jsonify({"error": "Prediction models are unavailable."}), 503

    try:
        result = predict(model, payload)
        result["model"] = model_name

        # Persist prediction to database
        pred_id = log_prediction(
            model_name=model_name,
            input_params=result["inputs"],
            prediction=result["prediction"],
            probability=result["probability"],
            risk_label=result["risk_label"],
        )
        result["prediction_id"] = pred_id
        result["disclaimer"] = "Educational ML decision support only; not a clinical diagnosis or treatment recommendation."
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Internal prediction error: {exc}"}), 500


@app.post("/api/sensitivity")
def sensitivity():
    """Live parameter sensitivity what-if analysis on selected model."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    model_name = payload.get("model", DEFAULT_MODEL)
    if model_name not in MODEL_REGISTRY:
        model_name = next(iter(MODEL_REGISTRY.keys()), DEFAULT_MODEL)

    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        return jsonify({"error": "Selected model is unavailable."}), 503

    feature = payload.get("feature")
    baseline = payload.get("baseline")
    values = payload.get("values")

    if not isinstance(feature, str) or not isinstance(baseline, dict) or not isinstance(values, list):
        return jsonify({"error": "feature, baseline, and values are required."}), 400

    if len(values) > 30:
        return jsonify({"error": "A maximum of 30 what-if values is allowed."}), 400

    try:
        clean_baseline = validate_patient_input(baseline)
        clean_values = [float(v) for v in values]
        results = sensitivity_analysis(model, clean_baseline, feature, clean_values)
        return jsonify({
            "model": model_name,
            "feature": feature,
            "results": results,
            "disclaimer": "Sensitivity curves describe model behavioral dynamics; not clinical causation.",
        })
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/history")
def history():
    """Retrieve recent patient assessment history from persistent storage."""
    try:
        limit = min(int(request.args.get("limit", 30)), 100)
        records = get_recent_predictions(limit=limit)
        return jsonify({"history": records, "count": len(records)})
    except Exception as exc:
        return jsonify({"error": f"History could not be retrieved: {exc}"}), 500


@app.get("/api/experiments")
def experiments():
    """Retrieve verified machine learning experiment records."""
    try:
        benchmark_info = load_benchmark(MODEL_DIR)
        return jsonify({
            "dataset": benchmark_info.get("dataset"),
            "dataset_rows": benchmark_info.get("dataset_rows"),
            "selected_model": benchmark_info.get("selected_model"),
            "experiments": benchmark_info.get("models", []),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/report", methods=["GET", "POST"])
@app.route("/api/export/pdf", methods=["GET", "POST"])
def export_pdf():
    """Generate and return a professional clinical intelligence PDF report."""
    prediction_data = {}
    if request.method == "POST":
        prediction_data = request.get_json(silent=True) or {}
    else:
        # Default baseline if GET
        prediction_data = {
            "model": DEFAULT_MODEL,
            "prediction": 0,
            "probability": 0.28,
            "risk_level": "Low",
            "inputs": {
                "Pregnancies": 2,
                "Glucose": 110,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 24.5,
                "DiabetesPedigreeFunction": 0.35,
                "Age": 28,
            },
        }

    try:
        bench_info = load_benchmark(MODEL_DIR).get("models", [])
    except Exception:
        bench_info = []

    model_name = prediction_data.get("model", DEFAULT_MODEL)
    model = MODEL_REGISTRY.get(model_name)
    importance_info = global_feature_importance(model, FEATURES) if model else []

    pdf_bytes = generate_pdf_report(
        prediction_data=prediction_data,
        benchmark_data=bench_info,
        importance_data=importance_info,
    )

    filename = f"smart-healthcare-report-{model_name.lower().replace(' ', '-')}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/export/csv")
def export_csv():
    """Export benchmark evaluation metrics as CSV."""
    try:
        bench = load_benchmark(MODEL_DIR).get("models", [])
        output = io.StringIO()
        if bench:
            writer = csv.DictWriter(output, fieldnames=list(bench[0].keys()))
            writer.writeheader()
            writer.writerows(bench)
        csv_data = output.getvalue()
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=model-benchmark-metrics.csv"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/export/json")
def export_json():
    """Export complete canonical evaluation artifact as JSON."""
    eval_path = MODEL_DIR / "evaluation.json"
    if not eval_path.exists():
        return jsonify({"error": "Artifacts unavailable."}), 503
    return Response(
        eval_path.read_text(encoding="utf-8"),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=smart-healthcare-evaluation.json"},
    )


@app.get("/")
def home():
    """Render main application shell."""
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
