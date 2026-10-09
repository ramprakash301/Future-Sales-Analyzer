
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

data_path = (
    PROJECT_ROOT
    / "module1"
    / "data"
    / "processed"
    / "ecommerce_sales_cleaned.csv"
)

# Save new results in the same folder as this Python file
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 3. Select features available before/during order placement
numeric_features = [
    "price",
    "discount",
    "quantity",
    "customer_age"
]

categorical_features = [
    "category",
    "payment_method",
    "region",
    "customer_gender"
]

# Use order date to derive information known at order time
df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df["order_month"] = df["order_date"].dt.month
df["order_day_of_week"] = df["order_date"].dt.dayofweek

numeric_features.extend([
    "order_month",
    "order_day_of_week"
])

# Check required columns
required_columns = (
    numeric_features
    + categorical_features
    + ["returned"]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove rows with missing target values
df = df.dropna(subset=["returned"]).copy()

# 4. Prepare input and target
X = df[numeric_features + categorical_features].copy()

# Target: Yes = returned, No = not returned
y = df["returned"].map({
    "No": 0,
    "Yes": 1
})

# Remove any unexpected target labels
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

# Fill missing numeric values using median
for column in numeric_features:
    X[column] = X[column].fillna(X[column].median())

# Fill missing categorical values
for column in categorical_features:
    X[column] = X[column].fillna("Unknown").astype(str)

print("\nFeatures selected:", list(X.columns))
print("\nTarget distribution:")
print(y.value_counts())

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# 7. Build classification model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# 8. Train the model
print("\nTraining Return-Risk Prediction Model...")
model.fit(X_train, y_train)

# 9. Predict on the test dataset
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# 10. Evaluate the model
print("\n========== MODEL EVALUATION ==========")

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Returned", "Returned"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, y_probability), 4)
)

print(
    "PR-AUC (Average Precision):",
    round(average_precision_score(y_test, y_probability), 4)
)

# 11. Save test-set predictions
results = pd.DataFrame({
    "actual_returned": y_test.map({
        0: "No",
        1: "Yes"
    }),
    "predicted_returned": pd.Series(
        y_pred,
        index=y_test.index
    ).map({
        0: "No",
        1: "Yes"
    }),
    "return_risk_score": y_probability.round(4)
})

# Keep order IDs for reference
results.insert(
    0,
    "order_id",
    df.loc[results.index, "order_id"].values
)

output_path = OUTPUT_DIR / "return_risk_predictions.csv"
results.to_csv(output_path, index=False)

print("\nPredictions saved to:")
print(output_path)

print("\nReturn-Risk Prediction Model completed!")

import joblib

# Save the trained return-risk model
model_path = OUTPUT_DIR / "return_risk_model.pkl"
joblib.dump(model, model_path)

print("Return-risk model saved to:")
print(model_path)
