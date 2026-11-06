import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Load the preprocessed data saved in Step 2
X_train = pd.read_csv("Data Folder/X_train.csv")
y_train = pd.read_csv("Data Folder/y_train.csv").squeeze()  # convert 1-column DataFrame to Series

print("Training data loaded:", X_train.shape)

# 2. Define the models we want to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# 3. Train each model and save it to disk
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)  # learn patterns from training data

    # Save the trained model as a .pkl file (used later for comparison and deployment)
    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"Data Folder/{file_name}")
    print(f"Saved: {file_name}")

# 4. Save the column names so the API can use the same order later (needed in deployment step)
joblib.dump(list(X_train.columns), "Data Folder/columns.pkl")

print("\nAll models trained and saved successfully")