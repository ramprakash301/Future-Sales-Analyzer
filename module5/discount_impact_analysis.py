import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "module1/data/processed/ecommerce_sales_cleaned.csv"
)

print("Dataset loaded successfully.")

# Calculate discount amount
df["discount_amount"] = df["gross_sales"] - df["net_sales"]

# Calculate discount impact percentage
df["discount_impact_percent"] = (
    df["discount_amount"] / df["gross_sales"]
) * 100

# Overall discount analysis
total_gross_sales = df["gross_sales"].sum()
total_net_sales = df["net_sales"].sum()
total_discount_amount = df["discount_amount"].sum()

overall_discount_impact = (
    total_discount_amount / total_gross_sales
) * 100

print("\nDiscount Impact Analysis")
print("------------------------")
print("Total Gross Sales:", round(total_gross_sales, 2))
print("Total Net Sales:", round(total_net_sales, 2))
print("Total Discount Amount:", round(total_discount_amount, 2))
print("Overall Discount Impact:", round(overall_discount_impact, 2), "%")

# Discount-wise analysis
discount_analysis = (
    df.groupby("discount_percent")
    .agg(
        orders=("order_id", "count"),
        gross_sales=("gross_sales", "sum"),
        net_sales=("net_sales", "sum"),
        discount_amount=("discount_amount", "sum")
    )
    .reset_index()
)

discount_analysis["discount_impact_percent"] = (
    discount_analysis["discount_amount"]
    / discount_analysis["gross_sales"]
) * 100

print("\nDiscount-wise Sales Analysis:")
print(discount_analysis)

# Category-wise discount impact
category_discount = (
    df.groupby("category")
    .agg(
        gross_sales=("gross_sales", "sum"),
        net_sales=("net_sales", "sum"),
        discount_amount=("discount_amount", "sum")
    )
    .reset_index()
)

category_discount["discount_impact_percent"] = (
    category_discount["discount_amount"]
    / category_discount["gross_sales"]
) * 100

print("\nCategory-wise Discount Impact:")
print(category_discount)

# Save outputs
discount_analysis.to_csv(
    "module5_discount_impact_analysis.csv",
    index=False
)

category_discount.to_csv(
    "module5_category_discount_analysis.csv",
    index=False
)

print("\nDiscount impact analysis saved successfully.")
