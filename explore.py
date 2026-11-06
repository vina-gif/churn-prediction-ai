import pandas as pd

df = pd.read_csv("Data Folder/telco_churn.csv")

print(df.shape)
print(df.head())
print(df["Churn"].value_counts())