from .data_pipeline import FEATURES,TARGET
def summary_statistics(df): return df[FEATURES].describe().T.reset_index(names="feature")
def correlation_matrix(df): return df[FEATURES+[TARGET]].corr(numeric_only=True)
def outcome_summary(df):
 c=df[TARGET].value_counts().sort_index(); r=__import__('pandas').DataFrame({"outcome":c.index.astype(int),"count":c.values}); r["percentage"]=r["count"]/r["count"].sum()*100; return r
def feature_by_outcome(df): return df.groupby(TARGET)[FEATURES].mean().T.reset_index(names="feature")
