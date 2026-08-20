from .data_pipeline import FEATURES
def quality_summary(df):
    missing=df[FEATURES].isna().sum(); duplicates=int(df.duplicated().sum()); numeric=df[FEATURES].select_dtypes(include="number"); finite=int(numeric.replace([float("inf"),float("-inf")],float("nan")).notna().sum().sum()); total=int(numeric.size)
    return {"rows":int(len(df)),"features":len(FEATURES),"missing_values":int(missing.sum()),"missing_by_feature":{k:int(v) for k,v in missing.items()},"duplicate_rows":duplicates,"finite_value_ratio":float(finite/total) if total else 1.0}
