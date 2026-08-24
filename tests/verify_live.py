"""Live Production Verification Script."""
import json
import urllib.request

BASE_URL = "https://smart-healthcare-system.onrender.com"


def run_live_verification():
    print(f"Connecting to live production endpoint: {BASE_URL}")

    # 1. Health
    req = urllib.request.Request(f"{BASE_URL}/api/health", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        health = json.loads(res.read().decode())
        print(f"1. Health Check [HTTP {res.status}]: {health}")

    # 2. Available Models
    req = urllib.request.Request(f"{BASE_URL}/api/models", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        models = json.loads(res.read().decode())
        print(f"2. Models Available: {models['models']}")

    # 3. Model Predictions (All 3 Models)
    patient = {
        "Pregnancies": 3,
        "Glucose": 150,
        "BloodPressure": 74,
        "SkinThickness": 28,
        "Insulin": 140,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.6,
        "Age": 45,
    }
    for m in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        payload = json.dumps({"model": m, **patient}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as res:
            pred = json.loads(res.read().decode())
            print(f"3. Prediction [{m}]: Prediction={pred['prediction']}, Label={pred['risk_label']}, Risk={pred['risk_percentage']}%")

    # 4. Evaluation Evidence
    req = urllib.request.Request(f"{BASE_URL}/api/evaluation?model=Random%20Forest", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        eval_data = json.loads(res.read().decode())
        auc = eval_data["models"]["Random Forest"]["metrics"]["roc_auc"]
        acc = eval_data["models"]["Random Forest"]["metrics"]["accuracy"]
        print(f"4. Canonical Evaluation: Selected={eval_data['selected_model']}, RF ROC-AUC={auc}, Accuracy={acc}")

    # 5. Explainability
    req = urllib.request.Request(f"{BASE_URL}/api/explainability?model=Random%20Forest", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        exp = json.loads(res.read().decode())
        print(f"5. Explainability Top Features: {exp['importance'][:3]}")

    # 6. Sensitivity Simulation
    sens_payload = json.dumps({
        "model": "Random Forest",
        "feature": "Glucose",
        "baseline": patient,
        "values": [80, 120, 160, 200],
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/sensitivity",
        data=sens_payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as res:
        sens = json.loads(res.read().decode())
        points = [(r["value"], str(r["risk_percentage"]) + "%") for r in sens["results"]]
        print(f"6. Sensitivity What-If Points: {points}")

    # 7. Benchmark
    req = urllib.request.Request(f"{BASE_URL}/api/benchmark", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        bm = json.loads(res.read().decode())
        print(f"7. Benchmark Tested Models: {[x['model'] for x in bm['models']]}")

    # 8. ReportLab PDF Export
    req = urllib.request.Request(f"{BASE_URL}/api/export/pdf", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        pdf_bytes = res.read()
        print(f"8. ReportLab PDF Generated: {len(pdf_bytes)} bytes, Content-Type: {res.headers.get('Content-Type')}")

    print("\n=======================================================")
    print("ALL PRODUCTION CHECKS VERIFIED SUCCESSFULLY ON LIVE APP")
    print("=======================================================")


if __name__ == "__main__":
    run_live_verification()
