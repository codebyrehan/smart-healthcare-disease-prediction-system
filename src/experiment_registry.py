import json
from pathlib import Path
from datetime import datetime,timezone
REGISTRY_PATH=Path("artifacts/experiments.json")
def load_experiments(registry_path=REGISTRY_PATH):
 if not registry_path.exists(): return []
 data=json.loads(registry_path.read_text());
 if not isinstance(data,list): raise ValueError("Experiment registry must contain a JSON list.")
 return data
def record_experiment(*,model_name,metrics,hyperparameters,dataset_version,feature_names,random_state=42,registry_path=REGISTRY_PATH):
 records=load_experiments(registry_path); exp={"experiment_id":f"exp-{len(records)+1:04d}","timestamp_utc":datetime.now(timezone.utc).isoformat(),"model_name":model_name,"dataset_version":dataset_version,"feature_names":list(feature_names),"hyperparameters":hyperparameters,"metrics":{str(k):float(v) for k,v in metrics.items()},"random_state":int(random_state)}; registry_path.parent.mkdir(parents=True,exist_ok=True); records.append(exp); registry_path.write_text(json.dumps(records,indent=2)); return exp
