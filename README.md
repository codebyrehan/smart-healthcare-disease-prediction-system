# Smart Healthcare System for Disease Prediction

A comprehensive machine learning project for predicting diabetes using the PIMA Indians Diabetes Dataset.

## 📋 Project Overview

This project implements a **Smart Healthcare System** that uses machine learning algorithms to predict whether a patient has diabetes based on medical diagnostic measurements. The system includes data preprocessing, exploratory data analysis (EDA), and comparison of multiple classification models.

**Student:** MOHD REHAN  
**Roll No:** 25SCS1003003208  
**Section:** 1CSE26  
**Submitted to:** MS. ISHU CHAUDHARY

---

## 🎯 Objectives

- Load and preprocess the PIMA Indians Diabetes Dataset
- Perform comprehensive Exploratory Data Analysis (EDA)
- Build and train multiple machine learning models
- Evaluate and compare model performance
- Generate visualizations for insights
- Provide a production-ready healthcare prediction system

---

## 📊 Dataset

**Source:** PIMA Indians Diabetes Dataset  
**Samples:** 768 records  
**Features:** 8 medical diagnostic measurements  
**Target:** Diabetes outcome (Binary: 0 = Non-Diabetic, 1 = Diabetic)

### Features:
- **Pregnancies:** Number of times pregnant
- **Glucose:** Plasma glucose concentration
- **BloodPressure:** Diastolic blood pressure
- **SkinThickness:** Triceps skin fold thickness
- **Insulin:** 2-Hour serum insulin level
- **BMI:** Body Mass Index
- **DiabetesPedigreeFunction:** Diabetes pedigree function
- **Age:** Age of patient

---

## 🏗️ Project Structure

```
smart-healthcare-disease-prediction-system/
├── smart_healthcare.py          # Main project script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── data/
│   └── PIMA_Diabetes_Dataset.xlsx # Input dataset
├── outputs/                      # Generated visualizations
│   ├── fig1_distributions.png
│   ├── fig2_heatmap.png
│   ├── fig3_class_boxplots.png
│   ├── fig4_pairplot.png
│   ├── fig5_confusion_matrices.png
│   ├── fig6_roc_curves.png
│   ├── fig7_model_comparison.png
│   └── fig8_feature_importance.png
└── models/                       # Trained models (optional)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/smart-healthcare-disease-prediction-system.git
   cd smart-healthcare-disease-prediction-system
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Run the main project script:

```bash
python smart_healthcare.py
```

**What the script does:**
1. Loads the PIMA Diabetes Dataset (with online/offline fallback)
2. Performs data cleaning and preprocessing
3. Conducts Exploratory Data Analysis
4. Trains 3 machine learning models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. Evaluates models using multiple metrics
6. Generates 8 comprehensive visualizations
7. Automatically opens all figures in your default image viewer

---

## 📈 Models Implemented

### 1. Logistic Regression
- Linear classification model
- Good for binary classification
- Fast and interpretable

### 2. Decision Tree Classifier
- Tree-based model with max depth of 5
- Handles non-linear relationships
- Easy to visualize and interpret

### 3. Random Forest Classifier
- Ensemble model with 100 trees
- Reduces overfitting through averaging
- Provides feature importance rankings

---

## 📊 Evaluation Metrics

The project evaluates models using:

- **Accuracy:** Percentage of correct predictions
- **ROC-AUC Score:** Area under the ROC curve (0.5-1.0)
- **Confusion Matrix:** True/False Positives and Negatives
- **Classification Report:** Precision, Recall, F1-Score
- **ROC Curves:** Visual comparison of all models

---

## 📈 Key Findings

The analysis generates visualizations for:

1. **Feature Distributions** - Distribution of each feature by outcome
2. **Correlation Heatmap** - Relationships between features
3. **Class Distribution** - Balance of diabetic vs non-diabetic cases
4. **Pairplot** - Interactions between key features
5. **Confusion Matrices** - Performance breakdown for each model
6. **ROC Curves** - Model comparison on different thresholds
7. **Model Comparison** - Accuracy and AUC comparison
8. **Feature Importance** - Most important features in Random Forest

---

## 📋 Data Preprocessing Steps

1. **Missing Value Handling:** Replace impossible zeros with NaN
2. **Imputation:** Fill missing values with median
3. **Feature Scaling:** StandardScaler normalization
4. **Train-Test Split:** 80-20 split with stratification
5. **Data Validation:** Ensure data quality before modeling

---

## 🔍 Technical Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning models |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualization |
| **Git** | Version control |

---

## 📝 Results Summary

The script outputs:
- Detailed console logs for each step
- Accuracy and AUC scores for all models
- Classification reports with precision/recall
- 8 professional visualizations
- Best performing model recommendation

---

## ⚙️ Configuration

You can modify the following parameters in the script:

```python
# Model parameters
DecisionTreeClassifier(max_depth=5, random_state=42)
RandomForestClassifier(n_estimators=100, random_state=42)
LogisticRegression(max_iter=1000, random_state=42)

# Train-test split
test_size=0.2, random_state=42, stratify=y

# Visualization style
plt.rcParams.update({...})
```

---

## 🎨 Visualizations

All generated figures are automatically displayed and saved in the `outputs/` directory:
- PNG format (150 dpi)
- Professional styling with consistent color palette
- High-quality for reports and presentations

---

## 🔗 Dataset Source

The PIMA Indians Diabetes Dataset is sourced from:
- **Original:** UCI Machine Learning Repository
- **Current URL:** GitHub raw content
- **Fallback:** Synthetic data generation (for offline mode)

---

## 📝 License

This is a student project submitted for coursework. Feel free to use for educational purposes.

---

## 📧 Contact

**Author:** MOHD REHAN  
**Email:** codexrehan@gmail.com  
**Roll No:** 25SCS1003003208

---

## 🙏 Acknowledgments

- Guided by: **MS. ISHU CHAUDHARY**
- Section: **1CSE26**
- Institution: **IILM University Greater Noida**

---

**Last Updated:** April 2026
