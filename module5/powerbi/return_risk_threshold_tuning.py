
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# 1. Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "module1"
    / "data"
    / "processed"
    / "ecommerce_sales_cleaned.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df["order_month"] = df["order_date"].dt.month
df["order_day_of_week"] = df["order_date"].dt.dayofweek

numeric_features = [
    "price",
    "discount",
    "quantity",
    "customer_age",
    "order_month",
    "order_day_of_week"
]

categorical_features = [
    "category",
    "payment_method",
    "region",
    "customer_gender"
]

X = df[numeric_features + categorical_features].copy()
y = df["returned"].map({"No": 0, "Yes": 1})

valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

# 3. Split off the final test set first
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

# 5. Build the Extra Trees model
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    ("numeric", "passthrough", numeric_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("Training Extra Trees model...")
model.fit(X_train, y_train)

# 6. Get validation scores
val_scores = model.predict_proba(X_val)[:, 1]

# Test different thresholds on validation data only
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

validation_results = []

for threshold in thresholds:
    val_predictions = (val_scores >= threshold).astype(int)

    validation_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, val_predictions, zero_division=0
        )
    })

validation_df = pd.DataFrame(validation_results)

print("\n===== VALIDATION THRESHOLD COMPARISON =====")
print(validation_df.round(4).to_string(index=False))

# Choose threshold with highest validation F1-score
best_row = validation_df.loc[
    validation_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nSelected threshold:", best_threshold)
print("Validation F1-score:", round(best_row["f1_score"], 4))

# 7. Evaluate selected threshold on untouched test data
test_scores = model.predict_proba(X_test)[:, 1]
test_predictions = (test_scores >= best_threshold).astype(int)

print("\n===== FINAL TEST RESULTS =====")
print("Threshold:", best_threshold)
print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "F1-score:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_scores), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

# 8. Save validation comparison and final test predictions
validation_df.to_csv(
    OUTPUT_DIR / "return_risk_threshold_results.csv",
    index=False
)

test_results = pd.DataFrame({
    "actual_returned": y_test.map({0: "No", 1: "Yes"}),
    "predicted_returned": pd.Series(
        test_predictions, index=y_test.index
    ).map({0: "No", 1: "Yes"}),
    "return_risk_score": test_scores.round(4),
    "threshold_used": best_threshold
})

test_results.insert(
    0,
    "order_id",
    df.loc[test_results.index, "order_id"].values
)

test_results.to_csv(
    OUTPUT_DIR / "return_risk_threshold_predictions.csv",
    index=False
)

print("\nFiles saved:")
print(OUTPUT_DIR / "return_risk_threshold_results.csv")
print(OUTPUT_DIR / "return_risk_threshold_predictions.csv")

print("\nThreshold tuning completed!")
