
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

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

print("Dataset loaded:", df.shape)

# 3. Create date features
df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
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

# 4. Prepare input and target
X = df[numeric_features + categorical_features].copy()
y = df["returned"].map({"No": 0, "Yes": 1})

valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Preprocess features
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    ("numeric", "passthrough", numeric_features)
])

# 7. Train Extra Trees model
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

print("Training model...")
model.fit(X_train, y_train)

# 8. Extract feature importance
encoder = (
    model.named_steps["preprocessor"]
    .named_transformers_["categorical"]
)

encoded_names = encoder.get_feature_names_out(
    categorical_features
)

all_feature_names = list(encoded_names) + numeric_features

importance_values = (
    model.named_steps["classifier"].feature_importances_
)

importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importance_values
})

# 9. Group encoded categories into original feature groups
importance_df["feature_group"] = (
    importance_df["feature"]
    .str.split("_")
    .str[0]
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n===== TOP 15 ENCODED FEATURES =====")
print(
    importance_df[
        ["feature", "importance"]
    ].head(15).round(5).to_string(index=False)
)

# Aggregate importance by original feature group
group_importance = (
    importance_df.groupby("feature_group")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

group_importance.columns = [
    "feature_group",
    "total_importance"
]

print("\n===== FEATURE GROUP IMPORTANCE =====")
print(group_importance.round(5).to_string(index=False))

# 10. Save reports
importance_df.to_csv(
    OUTPUT_DIR / "return_risk_feature_importance.csv",
    index=False
)

group_importance.to_csv(
    OUTPUT_DIR / "return_risk_feature_group_importance.csv",
    index=False
)

print("\nFeature importance reports saved in:")
print(OUTPUT_DIR)
