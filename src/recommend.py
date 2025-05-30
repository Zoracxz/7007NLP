import pandas as pd

df = pd.read_pickle("data/final_predictions.pkl")

# 获取所有疾病名称，供下拉框使用
def get_all_conditions():
    return sorted(df["condition"].unique())

# 给定疾病名，推荐 Top 10 药物
def get_recommendations(condition_name):
    filtered = df[df["condition"].str.lower() == condition_name.lower()]
    if filtered.empty:
        return ["No matching drugs found. Please check the condition name."]
    top_drugs = (
        filtered.sort_values(by="total_pred", ascending=False)
        .head(10)["drugName"]
        .tolist()
    )
    return top_drugs
