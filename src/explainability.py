import numpy as np
from .data_pipeline import FEATURES
def global_feature_importance(model,feature_names=None):
 if feature_names and list(feature_names)!=FEATURES: raise ValueError("feature_names must match project schema")
 est=model.named_steps.get("model",model) if hasattr(model,"named_steps") else model
 values=getattr(est,"feature_importances_",None)
 if values is None:
  coef=getattr(est,"coef_",None); values=np.abs(np.asarray(coef)).ravel() if coef is not None else None
 if values is None or len(values)!=len(FEATURES): return []
 values=np.asarray(values,float); total=values.sum(); values=values/total if total else values
 return [{"feature":f,"importance":float(v)} for f,v in sorted(zip(FEATURES,values),key=lambda x:x[1],reverse=True)]
