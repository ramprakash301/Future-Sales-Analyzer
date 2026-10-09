import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "module1/data/processed/ecommerce_sales_cleaned.csv"
)

print("Dataset loaded successfully.")
print("Total Orders:", len(df))

# Count returned and non-returned orders
returned_orders = (df["returned"] == "Yes").sum()
non_returned_orders = (df["returned"] == "No").sum()
total_orders = len(df)

# Calculate return rate
return_rate = (returned_orders / total_orders) * 100

print("\nReturn Product Analysis")
print("-----------------------")
print("Total Orders:", total_orders)
print("Returned Orders:", returned_orders)
print("Non-Returned Orders:", non_returned_orders)
print("Return Rate:", round(return_rate, 2), "%")

# Category-wise return analysis
category_returns = (
    df.groupby("category")["returned_flag"]
    .sum()
    .reset_index(name="returned_orders")
    .sort_values("returned_orders", ascending=False)
)

print("\nCategory-wise Returned Products:")
print(category_returns)

# Save output
category_returns.to_csv(
    "module5_category_return_analysis.csv",
    index=False
)

print("\nReturn analysis saved successfully.")