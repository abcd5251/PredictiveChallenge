import pandas as pd

df = pd.read_csv("dataset.csv")



feature = list(df["weight_a"])

count = 0
for i in feature:
    if i <= 0.01:
        count += 1

print(count/ len(feature))