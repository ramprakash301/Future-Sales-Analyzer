
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Project paths
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

# 3. Create the same test split used in earlier scripts
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Separate training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

# 5. Build model pipeline
def build_model(classifier):
    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

models = {
    "Random Forest (0.50)": (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        0.50
    ),
    "Extra Trees (0.50)": (
        ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        0.50
    ),
    "Extra Trees (0.30)": (
        ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        0.30
    )
}

# 6. Train, predict and evaluate
report_rows = []
confusion_rows = []

for name, (classifier, threshold) in models.items():
    print(f"\nEvaluating {name}...")

    model = build_model(classifier)
    model.fit(X_train_full, y_train_full)

    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report_rows.append({
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_returned": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_returned": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_returned": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(y_test, scores),
        "roc_auc": roc_auc_score(y_test, scores)
    })

    confusion_rows.append({
        "model": name,
        "true_not_returned": tn,
        "false_returned_alerts": fp,
        "missed_returns": fn,
        "correctly_detected_returns": tp
    })

# 7. Display report
metrics_df = pd.DataFrame(report_rows).round(4)
confusion_df = pd.DataFrame(confusion_rows)

print("\n===== MODEL EVALUATION REPORT =====")
print(metrics_df.to_string(index=False))

print("\n===== CONFUSION MATRIX SUMMARY =====")
print(confusion_df.to_string(index=False))

# 8. Save reports
metrics_df.to_csv(
    OUTPUT_DIR / "return_risk_evaluation_metrics.csv",
    index=False
)

confusion_df.to_csv(
    OUTPUT_DIR / "return_risk_evaluation_confusion.csv",
    index=False
)

# 9. Create a readable text summary
summary_path = OUTPUT_DIR / "return_risk_evaluation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("RETURN-RISK MODEL EVALUATION REPORT\n")
    file.write("=" * 40 + "\n\n")
    file.write("Dataset records: 34,500\n")
    file.write("Test size: 20%\n")
    file.write("Target: returned (Yes/No)\n")
    file.write("Models: Random Forest and Extra Trees\n\n")
    file.write("METRICS\n")
    file.write(metrics_df.to_string(index=False))
    file.write("\n\nCONFUSION MATRIX SUMMARY\n")
    file.write(confusion_df.to_string(index=False))
    file.write(
        "\n\nNote: Threshold 0.30 prioritizes finding more returns "
        "but can generate more false alerts.\n"
    )
    file.write(
        "Scores are based on this dataset and split; they do not "
        "guarantee future performance.\n"
    )

print("\nReports saved in:")
print(OUTPUT_DIR)
print("\nEvaluation report completed!")
