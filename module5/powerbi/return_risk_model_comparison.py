
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "module1"
    / "data"
    / "processed"
    / "ecommerce_sales_cleaned.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)
df["order_month"] = df["order_date"].dt.month
df["order_day_of_week"] = df["order_date"].dt.dayofweek

numeric_features = [
    "price", "discount", "quantity", "customer_age",
    "order_month", "order_day_of_week"
]

categorical_features = [
    "category", "payment_method", "region", "customer_gender"
]

X = df[numeric_features + categorical_features].copy()
y = df["returned"].map({"No": 0, "Yes": 1})

valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def build_model(classifier):
    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", classifier)
    ])

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    model = build_model(classifier)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "precision_returned": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_returned": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_returned": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        )
    })

comparison = pd.DataFrame(results).round(4)

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "return_risk_model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print(OUTPUT_DIR / "return_risk_model_comparison.csv")
