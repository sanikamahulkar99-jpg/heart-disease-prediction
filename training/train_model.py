import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# -----------------------------
# Clean column names
# -----------------------------
df.columns = df.columns.str.strip().str.lower()


# -----------------------------
# Target column
# -----------------------------
target_column = "target"

if target_column not in df.columns:
    raise ValueError(
        f"'{target_column}' column not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# -----------------------------
# Features and target
# -----------------------------
X = df.drop(columns=[target_column])
y = df[target_column]


# -----------------------------
# Convert categorical columns
# -----------------------------
X = pd.get_dummies(X, drop_first=False)


# Save feature names
feature_columns = X.columns.tolist()


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Create model pipeline
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# Save model + feature columns
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

model_data = {
    "model": model,
    "features": feature_columns
}

joblib.dump(model_data, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)