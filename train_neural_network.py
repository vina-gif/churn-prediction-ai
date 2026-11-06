import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 1. Set random seeds so results are reproducible
tf.random.set_seed(42)
np.random.seed(42)

# 2. Load the preprocessed data from Step 2
X_train = pd.read_csv("Data Folder/X_train.csv").astype("float32")
X_test = pd.read_csv("Data Folder/X_test.csv").astype("float32")
y_train = pd.read_csv("Data Folder/y_train.csv").squeeze().astype("float32")
y_test = pd.read_csv("Data Folder/y_test.csv").squeeze().astype("float32")

# 3. Build the neural network
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),   # one input per feature
    layers.Dense(64, activation="relu"),       # hidden layer 1
    layers.Dropout(0.3),                       # switch off 30% neurons to reduce overfitting
    layers.Dense(32, activation="relu"),       # hidden layer 2
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),     # output: churn probability (0 to 1)
])

# 4. Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train (EarlyStopping stops training when validation loss stops improving)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# 6. Evaluate on the test data
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

nn_results = {
    "Model": "Neural Network (Keras)",
    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
    "Precision": round(precision_score(y_test, y_pred), 4),
    "Recall": round(recall_score(y_test, y_pred), 4),
    "F1 Score": round(f1_score(y_test, y_pred), 4),
    "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
}
print("\nNeural Network Results:")
for k, v in nn_results.items():
    print(f"{k}: {v}")

# 7. Add the neural network to the comparison table from Step 4
comparison = pd.read_csv("Data Folder/model_comparison.csv")
comparison = comparison[comparison["Model"] != nn_results["Model"]]
comparison = pd.concat([comparison, pd.DataFrame([nn_results])], ignore_index=True)
comparison = comparison.sort_values("ROC-AUC", ascending=False)
comparison.to_csv("Data Folder/model_comparison.csv", index=False)
print("\nUpdated Comparison:\n")
print(comparison.to_string(index=False))

# 8. Plot accuracy and loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("neural_network_training.png", dpi=150)

# 9. Save the trained neural network
model.save("Data Folder/neural_network.keras")
print("\nSaved: neural_network.keras and neural_network_training.png")