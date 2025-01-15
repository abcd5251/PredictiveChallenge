import pandas as pd 

df = pd.read_csv("result.csv")

result = pd.DataFrame()
result["id"] = df["id"]

min_pred = df["preds_cos"].min()
max_pred = df["preds_cos"].max()
result["pred"] = (df["preds_cos"] - min_pred) / (max_pred - min_pred)


result.to_csv("submit4.csv", index = False)

