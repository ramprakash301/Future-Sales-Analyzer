
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "module1"
    / "data"
    / "processed"
    / "ecommerce_sales_cleaned.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# Convert target into readable format
df["returned"] = df["returned"].astype(str).str.strip()

# Analyse return rate by category
category_analysis = (
    df.groupby("category")
    .agg(
        total_orders=("returned", "size"),
        returned_orders=(
            "returned",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

category_analysis["return_rate_percent"] = (
    category_analysis["returned_orders"]
    / category_analysis["total_orders"]
    * 100
).round(2)

category_analysis = category_analysis.sort_values(
    "return_rate_percent",
    ascending=False
)

print("\n===== RETURN RATE BY CATEGORY =====")
print(category_analysis.to_string(index=False))

category_analysis.to_csv(
    OUTPUT_DIR / "return_rate_by_category.csv",
    index=False
)

# Analyse return rate by discount
df["discount_group"] = (
    df["discount"].astype(str) + "%"
)

discount_analysis = (
    df.groupby("discount_group")
    .agg(
        total_orders=("returned", "size"),
        returned_orders=(
            "returned",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

discount_analysis["return_rate_percent"] = (
    discount_analysis["returned_orders"]
    / discount_analysis["total_orders"]
    * 100
).round(2)

print("\n===== RETURN RATE BY DISCOUNT =====")
print(discount_analysis.to_string(index=False))

discount_analysis.to_csv(
    OUTPUT_DIR / "return_rate_by_discount.csv",
    index=False
)

# Analyse return rate by payment method
payment_analysis = (
    df.groupby("payment_method")
    .agg(
        total_orders=("returned", "size"),
        returned_orders=(
            "returned",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

payment_analysis["return_rate_percent"] = (
    payment_analysis["returned_orders"]
    / payment_analysis["total_orders"]
    * 100
).round(2)

payment_analysis = payment_analysis.sort_values(
    "return_rate_percent",
    ascending=False
)

print("\n===== RETURN RATE BY PAYMENT METHOD =====")
print(payment_analysis.to_string(index=False))

payment_analysis.to_csv(
    OUTPUT_DIR / "return_rate_by_payment.csv",
    index=False
)

# Analyse return rate by region
region_analysis = (
    df.groupby("region")
    .agg(
        total_orders=("returned", "size"),
        returned_orders=(
            "returned",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

region_analysis["return_rate_percent"] = (
    region_analysis["returned_orders"]
    / region_analysis["total_orders"]
    * 100
).round(2)

region_analysis = region_analysis.sort_values(
    "return_rate_percent",
    ascending=False
)

print("\n===== RETURN RATE BY REGION =====")
print(region_analysis.to_string(index=False))

region_analysis.to_csv(
    OUTPUT_DIR / "return_rate_by_region.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Four CSV reports saved in:")
print(OUTPUT_DIR)
