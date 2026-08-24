# Smart Healthcare System for Disease Prediction (2026 Edition)

A portfolio-grade AI clinical decision-support and diabetes risk prediction platform featuring interactive 3D medical visualization, multi-model evaluation, global explainability, sensitivity simulation, PostgreSQL persistence, and ReportLab PDF reporting.

---

## 📋 Executive Overview

The **Smart Healthcare System** is a data-driven web platform that enables clinicians, researchers, and users to perform early diabetes risk assessment based on the **PIMA Indians Diabetes Dataset**.

- **Author:** MOHD REHAN (Roll No: 25SCS1003003208 | Section: 1CSE26)
- **Institution:** IILM University
- **Core Purpose:** Educational ML & Clinical Decision Support (Non-diagnostic)

---

## ✨ Key Capabilities & Architectural Highlights

1. **3D Medical AI Visualization**:
   - Interactive 3D WebGL / Canvas medical AI core with orbital particle nodes and responsive mouse parallax tilt.
   - Circular biometric risk ring with dynamic probability sweeps.

2. **Authoritative Multi-Model Pipeline (Single Source of Truth)**:
   - **Logistic Regression**: Interpretable linear baseline with Standard Scaler pipeline.
   - **Decision Tree**: Non-linear tree structure with controlled depth.
   - **Random Forest**: 300-tree ensemble with balanced class weighting (optimal ROC-AUC: `0.822`).

3. **Advanced Model Evaluation Workspace**:
   - Dynamic ROC Curves with area under curve calculations.
   - Precision-Recall (PR) Curves with PR-AUC metric.
   - Reliability Calibration Curves (Quantile Binned).
   - Interactive 2×2 Confusion Matrix with True/False Positive/Negative breakdown.
   - Threshold Decision Matrix ($0.1 \dots 0.9$) highlighting the optimal F1 threshold.

4. **Global Feature Explainability & Sensitivity Simulator**:
   - Relative feature importance rankings across all three models.
   - Live what-if simulator testing parameter sensitivity held against patient baselines.

5. **PostgreSQL Persistence & Idempotent Migrations**:
   - Production PostgreSQL storage for clinical assessments and training run audits.
   - Idempotent migration engine (`scripts/migrate_db.py`) enforcing `DATABASE_URL` in production without leaking credentials.

6. **Automated Clinical Dossier Reporting**:
   - Professional PDF exports powered by **ReportLab**.
   - Structured CSV and JSON exports consuming the canonical backend data structure.

---

## 🏗️ Project Architecture

```
smart-healthcare-disease-prediction-system/
├── app.py                           # Flask backend & canonical API endpoints
├── requirements.txt                 # Pinned dependencies (pandas, numpy, scikit-learn, joblib)
├── render.yaml                      # Render Blueprint specification
├── .python-version                  # Pinned Python runtime (3.11.11)
├── .gitignore                       # Git ignore rules
├── README.md                        # Documentation
├── data/
│   ├── PIMA_Diabetes_Dataset.xlsx   # Canonical PIMA dataset (Excel)
│   └── pima_diabetes.csv            # Canonical PIMA dataset (CSV)
├── models/                          # Trained model artifacts & evaluation JSON
│   ├── best_model.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── evaluation.json             # Canonical evaluation evidence
│   ├── metrics.json                # Comparative benchmark metrics
│   └── metadata.json               # Model metadata & dataset quality audit
├── scripts/
│   ├── migrate_db.py                # Idempotent database migration script
│   └── train_production_model.py    # Multi-model training and evaluation generator
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py             # Schema validation & zero imputation
│   ├── data_quality.py              # Data health & coverage metrics
│   ├── eda.py                       # Exploratory analytics helpers
│   ├── db.py                        # Database connection & persistence layer
│   ├── prediction.py                # Input validation & risk prediction
│   ├── explainability.py            # Global feature importance engine
│   ├── sensitivity.py               # What-if scenario simulation engine
│   ├── benchmark.py                 # Benchmark loader
│   └── reporting.py                 # ReportLab PDF & export generator
├── static/
│   ├── style.css                    # 2026 SaaS Dark Theme & Design System
│   ├── 3d-healthcare.css            # 3D core & biometric visualization styles
│   ├── 3d-healthcare.js             # 3D WebGL/Canvas medical core & parallax
│   ├── app.js                       # Prediction form & biometric signal display
│   └── dashboard.js                 # Evaluation workspace, SVG charts & history
└── templates/
    └── index.html                   # Command Center UI template
```

---

## 🚀 Setup & Local Execution

### 1. Clone & Environment Setup
```bash
git clone https://github.com/codebyrehan/smart-healthcare-disease-prediction-system.git
cd smart-healthcare-disease-prediction-system

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Database Migrations
```bash
python scripts/migrate_db.py
```

### 3. Train & Evaluate Models
```bash
python scripts/train_production_model.py
```

### 4. Start Local Server
```bash
python app.py
```
Access the application at `http://127.0.0.1:5000`.

---

## 🌐 Production Deployment (Render)

1. Connect the repository to **Render**.
2. Select **Web Service** or use `render.yaml`.
3. Configure the following Environment Variables in the Render dashboard:
   - `DATABASE_URL`: Your PostgreSQL connection string.
   - `PYTHON_VERSION`: `3.11.11`
   - `FLASK_ENV`: `production`
4. The build command will automatically execute:
   ```bash
   pip install -r requirements.txt && python scripts/migrate_db.py && python scripts/train_production_model.py
   ```
5. The start command will run Gunicorn:
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```

---

## ⚖️ Responsible AI & Ethical Scope

- **Decision Support Only:** This platform is designed for academic demonstration and research. Predictions do not constitute medical diagnoses or prescriptions.
- **Population Constraints:** Results are derived from the PIMA historical cohort and should be validated before applying to distinct demographic populations.
- **Consult Clinicians:** All clinical decisions must be confirmed by qualified medical professionals.
