"""End-to-End Test Suite for Smart Healthcare Platform."""
import io
import json
import unittest
from pathlib import Path
from app import app
from src.data_pipeline import FEATURES, TARGET, clean_features, load_dataset
from src.prediction import validate_patient_input, INPUT_RANGES


class TestSmartHealthcareSystem(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_01_dataset_integrity(self):
        df = load_dataset()
        self.assertEqual(len(df), 768, "Dataset must contain exactly 768 records.")
        self.assertEqual(set(df.columns), set(FEATURES + [TARGET]))
        clean_df, report = clean_features(df)
        self.assertEqual(len(clean_df), 768)
        self.assertFalse(clean_df[FEATURES].isna().any().any(), "No NaNs should remain after cleaning.")
        self.assertEqual(report["class_distribution"], {"0": 500, "1": 268})

    def test_02_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["models_loaded"])
        self.assertTrue(data["dataset_loaded"])
        self.assertEqual(len(data["available_models"]), 3)

    def test_03_models_endpoint(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("Random Forest", data["models"])
        self.assertIn("Logistic Regression", data["models"])
        self.assertIn("Decision Tree", data["models"])

    def test_04_prediction_all_three_models(self):
        patient_sample = {
            "Pregnancies": 3,
            "Glucose": 145,
            "BloodPressure": 75,
            "SkinThickness": 28,
            "Insulin": 130,
            "BMI": 31.5,
            "DiabetesPedigreeFunction": 0.52,
            "Age": 42,
        }

        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            payload = {"model": model_name, **patient_sample}
            res = self.client.post("/api/predict", json=payload)
            self.assertEqual(res.status_code, 200, f"Predict failed for {model_name}")
            data = res.get_json()
            self.assertEqual(data["model"], model_name)
            self.assertIn(data["prediction"], [0, 1])
            self.assertTrue(0.0 <= data["probability"] <= 1.0)
            self.assertIn("risk_level", data)
            self.assertIn("prediction_id", data)
            self.assertTrue(data["prediction_id"].startswith("pred-"))

    def test_05_prediction_validation_boundaries(self):
        # Missing feature
        res = self.client.post("/api/predict", json={"Pregnancies": 2})
        self.assertEqual(res.status_code, 400)

        # Value out of range
        res = self.client.post("/api/predict", json={
            "Pregnancies": 2, "Glucose": 999, "BloodPressure": 70, "SkinThickness": 20,
            "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.4, "Age": 30
        })
        self.assertEqual(res.status_code, 400)

    def test_06_evaluation_canonical_structure(self):
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            res = self.client.get(f"/api/evaluation?model={model_name}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["selected_model"], model_name)
            self.assertIn(model_name, data["models"])
            m = data["models"][model_name]

            # Verify canonical structure
            self.assertIn("metrics", m)
            self.assertIn("confusion_matrix", m)
            self.assertIn("roc_curve", m)
            self.assertIn("pr_curve", m)
            self.assertIn("calibration", m)
            self.assertIn("thresholds", m)
            self.assertIn("feature_importance", m)

            # Check metrics
            metrics = m["metrics"]
            self.assertTrue(metrics["accuracy"] > 0.65)
            self.assertTrue(metrics["roc_auc"] > 0.70)
            self.assertTrue(metrics["f1"] > 0.45)

    def test_07_explainability_all_models(self):
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            res = self.client.get(f"/api/explainability?model={model_name}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["model"], model_name)
            self.assertEqual(len(data["importance"]), len(FEATURES))
            total_imp = sum(item["importance"] for item in data["importance"])
            self.assertAlmostEqual(total_imp, 1.0, places=2)

    def test_08_sensitivity_simulation(self):
        payload = {
            "model": "Random Forest",
            "feature": "Glucose",
            "baseline": {
                "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
                "Insulin": 80, "BMI": 26.0, "DiabetesPedigreeFunction": 0.4, "Age": 30
            },
            "values": [80, 110, 140, 170, 200]
        }
        res = self.client.post("/api/sensitivity", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["feature"], "Glucose")
        self.assertEqual(len(data["results"]), 5)
        first_prob = data["results"][0]["probability"]
        last_prob = data["results"][-1]["probability"]
        self.assertGreater(last_prob, first_prob)

    def test_09_benchmark_and_metadata(self):
        res_bm = self.client.get("/api/benchmark")
        self.assertEqual(res_bm.status_code, 200)
        self.assertEqual(len(res_bm.get_json()["models"]), 3)

        res_meta = self.client.get("/api/metadata")
        self.assertEqual(res_meta.status_code, 200)
        self.assertEqual(res_meta.get_json()["features"], FEATURES)

    def test_10_eda_endpoint(self):
        res = self.client.get("/api/eda")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["rows"], 768)
        self.assertEqual(len(data["statistics"]), len(FEATURES))
        self.assertIn("correlation", data)
        self.assertIn("outcomes", data)

    def test_11_persistent_history(self):
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("history", data)
        self.assertGreaterEqual(data["count"], 1)

    def test_12_exports(self):
        # PDF Export
        res_pdf = self.client.get("/api/export/pdf")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, "application/pdf")
        self.assertTrue(len(res_pdf.data) > 2000)

        # CSV Export
        res_csv = self.client.get("/api/export/csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("accuracy", res_csv.get_data(as_text=True).lower())

        # JSON Export
        res_json = self.client.get("/api/export/json")
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("models", res_json.get_json())

    def test_13_home_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Smart Healthcare", html)
        self.assertIn("Predict. Understand.", html)
        self.assertIn("Prevent.", html)


if __name__ == "__main__":
    unittest.main()
