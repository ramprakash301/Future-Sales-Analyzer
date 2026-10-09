import pandas as pd

# Load cleaned dataset
df = pd.read_csv("module1/data/processed/ecommerce_sales_cleaned.csv")

# Convert order date to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Create month period
df["month_period"] = df["order_date"].dt.to_period("M")

# Calculate monthly net sales
monthly_sales = (
    df.groupby("month_period")["net_sales"]
    .sum()
    .reset_index()
)

# Calculate previous month's sales
monthly_sales["previous_month_sales"] = (
    monthly_sales["net_sales"].shift(1)
)

# Mathematical calculation: Sales Growth %
monthly_sales["sales_growth_percent"] = (
    (monthly_sales["net_sales"] -
     monthly_sales["previous_month_sales"])
    / monthly_sales["previous_month_sales"]
) * 100

# Display result
print("\nMonthly Sales Growth Analysis:")
print(monthly_sales)

# Save result
monthly_sales.to_csv(
    "module5_sales_growth_analysis.csv",
    index=False
)

print("\nSales Growth Analysis completed successfully.")
print("File saved as: module5_sales_growth_analysis.csv")