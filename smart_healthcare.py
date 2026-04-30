"""
========================================================
  SMART HEALTHCARE SYSTEM FOR DISEASE PREDICTION
  Student: MOHD REHAN | Roll No: 25SCS1003003208
  Section: 1CSE26 | Submitted to: MS. ISHU CHAUDHARY
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
import warnings
import os
import webbrowser
from pathlib import Path
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# OUTPUT DIRECTORY (works on Windows & Linux)
# ─────────────────────────────────────────────
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
saved_figures = []  # Track all saved figures

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})
PALETTE  = ['#2ECC71', '#E74C3C']
BLUE     = '#2980B9'
GREEN    = '#27AE60'
RED      = '#E74C3C'
ORANGE   = '#E67E22'

# ════════════════════════════════════════════
# 1. DATA COLLECTION
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 1: DATA COLLECTION")
print("="*55)

url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "master/pima-indians-diabetes.data.csv")
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

try:
    df = pd.read_csv(url, names=cols)
    print("✔  Dataset loaded from online source.")
except Exception:
    # Fallback: generate representative synthetic data
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        'Pregnancies':              np.random.randint(0,18,n),
        'Glucose':                  np.random.normal(120,32,n).clip(0,200).astype(int),
        'BloodPressure':            np.random.normal(69,19,n).clip(0,122).astype(int),
        'SkinThickness':            np.random.normal(20,16,n).clip(0,99).astype(int),
        'Insulin':                  np.random.normal(80,115,n).clip(0,846).astype(int),
        'BMI':                      np.random.normal(32,7,n).clip(0,68).round(1),
        'DiabetesPedigreeFunction': np.random.exponential(0.47,n).clip(0.08,2.42).round(3),
        'Age':                      np.random.randint(21,82,n),
        'Outcome':                  np.random.binomial(1,0.35,n),
    })
    print("✔  Dataset loaded from synthetic fallback (offline mode).")

print(f"\n   Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Diabetic (1) : {df['Outcome'].sum()}")
print(f"   Non-Diabetic : {(df['Outcome']==0).sum()}")
print("\n   First 5 rows:")
print(df.head().to_string())

# ════════════════════════════════════════════
# 2. DATA CLEANING & PREPROCESSING
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 2: DATA CLEANING & PREPROCESSING")
print("="*55)

# Replace biologically impossible zeros with NaN
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print(f"\n   Missing values after zero-replacement:")
print(df.isnull().sum().to_string())

# Impute with median (robust to outliers)
for col in zero_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"   '{col}' → imputed with median = {median_val:.2f}")

# Final check — drop any remaining NaN rows just in case
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print("\n   ✔  No missing values remain:", df.isnull().sum().sum())

# Feature / target split
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n   Train size : {X_train.shape[0]} samples")
print(f"   Test  size : {X_test.shape[0]} samples")
print("   ✔  Features scaled with StandardScaler")

# ════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*55)
print(df.describe().round(2).to_string())

# ── Figure 1: Distribution plots ────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Feature Distributions by Outcome\n(Green = Non-Diabetic | Red = Diabetic)",
             fontsize=15, fontweight='bold', y=1.01)

features = ['Pregnancies','Glucose','BloodPressure',
            'SkinThickness','Insulin','BMI',
            'DiabetesPedigreeFunction','Age']

for i, feat in enumerate(features):
    ax = axes[i//3][i%3]
    for outcome, color, label in zip([0,1], PALETTE, ['Non-Diabetic','Diabetic']):
        ax.hist(df[df['Outcome']==outcome][feat], bins=25,
                alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)

axes[2][2].axis('off')   # hide unused subplot
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, 'fig1_distributions.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig1_path)
plt.close()
print("\n   ✔  Fig 1 saved: Feature Distributions")

# ── Figure 2: Correlation Heatmap ────────────
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size':9}, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap of All Features', fontsize=14, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, 'fig2_heatmap.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig2_path)
plt.close()
print("   ✔  Fig 2 saved: Correlation Heatmap")

# ── Figure 3: Outcome pie + boxplots ─────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle('Class Distribution & Key Feature Analysis', fontsize=15, fontweight='bold')

counts = df['Outcome'].value_counts()
axes[0].pie(counts, labels=['Non-Diabetic','Diabetic'],
            autopct='%1.1f%%', colors=PALETTE,
            startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Class Distribution')

key_feats = ['Glucose','BMI','Age','Insulin']
for idx, feat in enumerate(key_feats):
    ax = axes[idx+1]
    sns.boxplot(x='Outcome', y=feat, data=df, palette=PALETTE, ax=ax,
                flierprops={'marker':'o','alpha':0.4,'markersize':3})
    ax.set_title(f'{feat} vs Outcome')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Non-\nDiabetic','Diabetic'])
    ax.set_xlabel('')

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'fig3_class_boxplots.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig3_path)
plt.close()
print("   ✔  Fig 3 saved: Class Distribution & Boxplots")

# ── Figure 4: Pairplot (Glucose, BMI, Age, DPF) ──
pair_df = df[['Glucose','BMI','Age','DiabetesPedigreeFunction','Outcome']].copy()
pair_df['Outcome'] = pair_df['Outcome'].map({0:'Non-Diabetic', 1:'Diabetic'})
g = sns.pairplot(pair_df, hue='Outcome', palette={'Non-Diabetic':GREEN,'Diabetic':RED},
                 diag_kind='kde', plot_kws={'alpha':0.5,'s':20})
g.fig.suptitle('Pairplot: Key Features', y=1.02, fontsize=14, fontweight='bold')
fig4_path = os.path.join(OUTPUT_DIR, 'fig4_pairplot.png')
plt.savefig(fig4_path, dpi=120, bbox_inches='tight')
saved_figures.append(fig4_path)
plt.close()
print("   ✔  Fig 4 saved: Pairplot")

# ════════════════════════════════════════════
# 4. MODEL BUILDING & EVALUATION
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 4: MODEL BUILDING & EVALUATION")
print("="*55)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results   = {}
conf_mats = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:,1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, target_names=['Non-Diabetic','Diabetic'])

    results[name]   = {'accuracy': acc, 'auc': auc, 'proba': y_proba, 'pred': y_pred}
    conf_mats[name] = cm

    print(f"\n  ── {name} ──")
    print(f"     Accuracy : {acc*100:.2f}%")
    print(f"     ROC-AUC  : {auc:.4f}")
    print(f"     Confusion Matrix:\n{cm}")
    print(f"     Classification Report:\n{cr}")

# ── Figure 5: Confusion Matrices ─────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, (name, cm) in zip(axes, conf_mats.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Diabetic','Diabetic'],
                yticklabels=['Non-Diabetic','Diabetic'],
                linewidths=1, linecolor='white', annot_kws={'size':14})
    acc = results[name]['accuracy']
    ax.set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, 'fig5_confusion_matrices.png')
plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig5_path)
plt.close()
print("\n   ✔  Fig 5 saved: Confusion Matrices")

# ── Figure 6: ROC Curves ─────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = [BLUE, ORANGE, GREEN]

for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC = {res['auc']:.3f})")

ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
ax.fill_between([0,1],[0,1], alpha=0.05, color='grey')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
fig6_path = os.path.join(OUTPUT_DIR, 'fig6_roc_curves.png')
plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig6_path)
plt.close()
print("   ✔  Fig 6 saved: ROC Curves")

# ── Figure 7: Model Accuracy Comparison ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

names    = list(results.keys())
accs     = [results[n]['accuracy']*100 for n in names]
aucs     = [results[n]['auc'] for n in names]
bar_clrs = [BLUE, ORANGE, GREEN]

bars = axes[0].bar(names, accs, color=bar_clrs, edgecolor='white', width=0.5)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
axes[0].set_ylim(0, 110)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Accuracy Comparison')

bars2 = axes[1].bar(names, aucs, color=bar_clrs, edgecolor='white', width=0.5)
for bar, val in zip(bars2, aucs):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylim(0, 1.15)
axes[1].set_ylabel('AUC Score')
axes[1].set_title('ROC-AUC Comparison')

plt.tight_layout()
fig7_path = os.path.join(OUTPUT_DIR, 'fig7_model_comparison.png')
plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig7_path)
plt.close()
print("   ✔  Fig 7 saved: Model Comparison")

# ── Figure 8: Feature Importance (Random Forest) ──
rf_model   = models['Random Forest']
importances = rf_model.feature_importances_
feat_series = pd.Series(importances, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = [GREEN if v >= feat_series.median() else BLUE for v in feat_series]
bars = ax.barh(feat_series.index, feat_series.values, color=colors_fi, edgecolor='white')
for bar, val in zip(bars, feat_series.values):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
fig8_path = os.path.join(OUTPUT_DIR, 'fig8_feature_importance.png')
plt.savefig(fig8_path, dpi=150, bbox_inches='tight')
saved_figures.append(fig8_path)
plt.close()
print("   ✔  Fig 8 saved: Feature Importance")

# ════════════════════════════════════════════
# 5. SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("  FINAL SUMMARY")
print("="*55)
best = max(results, key=lambda n: results[n]['accuracy'])
print(f"\n  {'Model':<28} {'Accuracy':>10}  {'AUC':>8}")
print("  " + "-"*48)
for name in results:
    acc = results[name]['accuracy']*100
    auc = results[name]['auc']
    star = " ◄ BEST" if name==best else ""
    print(f"  {name:<28} {acc:>9.2f}%  {auc:>8.4f}{star}")

print(f"\n  Best Model  : {best}")
print(f"  Accuracy    : {results[best]['accuracy']*100:.2f}%")
print(f"  AUC Score   : {results[best]['auc']:.4f}")
print(f"\n  All outputs saved to '{OUTPUT_DIR}/' folder")
print("="*55 + "\n")

# ════════════════════════════════════════════
# 🎯 AUTO-OPEN ALL FIGURES IN BROWSER
# ════════════════════════════════════════════
print("\n🖼️  Opening all figures automatically...\n")

for fig_path in saved_figures:
    abs_path = str(Path(fig_path).absolute())
    webbrowser.open('file://' + abs_path)
    print(f"   ✔ Opened: {os.path.basename(fig_path)}")

print(f"\n✅ All {len(saved_figures)} figures opened in your default image viewer!")
print("="*55 + "\n")
