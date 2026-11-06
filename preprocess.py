import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Data load करो
df = pd.read_csv("Data Folder/telco_churn.csv")

# 2. TotalCharges को number में बदलो
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Missing values देखो और हटाओ
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
df = df.dropna()

# 4. customerID हटाओ
df = df.drop("customerID", axis=1)

# 5. Target (Churn) को 0/1 बनाओ
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Feature Engineering: नए columns बनाओ
df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    include_lowest=True,
)

service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

# 7. Text columns को numbers में बदलो (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# 8. Features (X) और Target (y) अलग करो
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 9. Train/Test split (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 10. Scaling
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargesPerMonth", "TotalServices"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
# Save the fitted scaler and numeric column list so the API can reuse them
joblib.dump(scaler, "Data Folder/scaler.pkl")
joblib.dump(num_cols, "Data Folder/num_cols.pkl")

# 11. Processed data save करो
X_train.to_csv("Data Folder/X_train.csv", index=False)
X_test.to_csv("Data Folder/X_test.csv", index=False)
y_train.to_csv("Data Folder/y_train.csv", index=False)
y_test.to_csv("Data Folder/y_test.csv", index=False)

print("\nPreprocessing पूरा हुआ ✅")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)