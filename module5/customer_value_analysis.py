import pandas as pd

# Load cleaned dataset
df = pd.read_csv("module1/data/processed/ecommerce_sales_cleaned.csv")

# Calculate customer-level values
customer_value = (
    df.groupby("customer_id")
    .agg(
        total_orders=("order_id", "count"),
        total_net_sales=("net_sales", "sum")
    )
    .reset_index()
)

# Mathematical calculation
customer_value["customer_value"] = (
    customer_value["total_net_sales"] /
    customer_value["total_orders"]
)

# Display top customers
top_customers = customer_value.sort_values(
    "customer_value",
    ascending=False
).head(10)

print("\nTop 10 Customers by Customer Value:")
print(top_customers)

# Save result
customer_value.to_csv(
    "module5_customer_value_analysis.csv",
    index=False
)

print("\nCustomer Value Analysis completed successfully.")
print("File saved as: module5_customer_value_analysis.csv")