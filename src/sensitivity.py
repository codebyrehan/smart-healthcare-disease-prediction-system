from .data_pipeline import FEATURES
def sensitivity_analysis(model,baseline,feature,values):
 if feature not in FEATURES: raise ValueError(f"Unsupported feature: {feature}")
 if set(baseline)!=set(FEATURES): raise ValueError("Baseline must contain exactly the validated model features.")
 rows=[]
 for value in values:
  c=dict(baseline); c[feature]=float(value); p=float(model.predict_proba([[c[n] for n in FEATURES]])[0][1]); rows.append({"value":float(value),"probability":p})
 return rows
