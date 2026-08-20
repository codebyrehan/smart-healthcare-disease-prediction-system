"""Data loading and validation for the PIMA diabetes dataset."""
from pathlib import Path
import pandas as pd
import numpy as np
FEATURES=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
TARGET="Outcome"
EXPECTED_COLUMNS=FEATURES+[TARGET]
IMPOSSIBLE_ZERO_COLUMNS=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
def load_dataset(path: str|Path="data/PIMA_Diabetes_Dataset.xlsx"):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Dataset not found at {p}.")
    df=pd.read_excel(p) if p.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(p)
    return validate_schema(df)
def validate_schema(df):
    missing=[c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Dataset is missing required columns: {missing}")
    r=df[EXPECTED_COLUMNS].copy()
    for c in EXPECTED_COLUMNS: r[c]=pd.to_numeric(r[c],errors="coerce")
    if r[TARGET].isna().any(): raise ValueError("Target contains missing or non-numeric values.")
    if (~r[TARGET].isin([0,1])).any(): raise ValueError("Outcome must contain only 0 and 1.")
    return r.drop_duplicates().reset_index(drop=True)
def clean_features(df):
    r=df.copy(); report={"rows_before":len(r),"duplicates_removed":0,"imputed":{}}
    before=len(r); r=r.drop_duplicates().reset_index(drop=True); report["duplicates_removed"]=before-len(r)
    for c in IMPOSSIBLE_ZERO_COLUMNS:
        n=int((r[c]==0).sum()); r.loc[r[c]==0,c]=np.nan; med=float(r[c].median()); r[c]=r[c].fillna(med); report["imputed"][c]={"count":n,"median":med}
    if r[FEATURES].isna().any().any(): raise ValueError("Unexpected missing feature values remain.")
    report["rows_after"]=len(r); return r,report
def data_quality_report(df):
    return {"rows":int(len(df)),"columns":int(df.shape[1]),"duplicates":int(df.duplicated().sum()),"missing_values":{k:int(v) for k,v in df.isna().sum().items()},"class_counts":{str(k):int(v) for k,v in df[TARGET].value_counts().sort_index().items()},"feature_ranges":{c:{"min":float(df[c].min()),"max":float(df[c].max())} for c in FEATURES}}
