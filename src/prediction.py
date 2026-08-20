import pandas as pd
from .data_pipeline import FEATURES
INPUT_RANGES={"Pregnancies":(0,30),"Glucose":(1,400),"BloodPressure":(1,250),"SkinThickness":(0,150),"Insulin":(0,1000),"BMI":(1,100),"DiabetesPedigreeFunction":(0,5),"Age":(1,120)}
def validate_patient_input(values):
 missing=[f for f in FEATURES if f not in values]
 if missing: raise ValueError(f"Missing required features: {missing}")
 out={}
 for f in FEATURES:
  try:v=float(values[f])
  except: raise ValueError(f"{f} must be numeric.")
  lo,hi=INPUT_RANGES[f]
  if not lo<=v<=hi: raise ValueError(f"{f} must be between {lo} and {hi}.")
  out[f]=v
 return out
def predict(model,values):
 v=validate_patient_input(values); p=float(model.predict_proba(pd.DataFrame([v],columns=FEATURES))[0,1]); y=int(p>=.5)
 return {"prediction":y,"label":"Higher likelihood" if y else "Lower likelihood","probability":p,"threshold":.5}
