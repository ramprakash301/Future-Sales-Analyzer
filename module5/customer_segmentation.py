import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "module1/data/processed/ecommerce_sales_cleaned.csv"
)

print("Dataset loaded successfully.")

# Customer-level sales aggregation
customer_sales = (
    df.groupby("customer_id")
    .agg(
        total_orders=("order_id", "count"),
        total_quantity=("quantity", "sum"),
        total_net_sales=("net_sales", "sum")
    )
    .reset_index()
)

# Customer segmentation based on total net sales
customer_sales["customer_segment"] = pd.qcut(
    customer_sales["total_net_sales"],
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)

print("\nCustomer Segmentation")
print("---------------------")
print(customer_sales.head())

# Segment summary
segment_summary = (
    customer_sales.groupby("customer_segment", observed=False)
    .agg(
        customers=("customer_id", "count"),
        total_sales=("total_net_sales", "sum"),
        total_orders=("total_orders", "sum")
    )
    .reset_index()
)

print("\nCustomer Segment Summary:")
print(segment_summary)

# Save outputs
customer_sales.to_csv(
    "module5_customer_segmentation.csv",
    index=False
)

segment_summary.to_csv(
    "module5_customer_segment_summary.csv",
    index=False
)

print("\nCustomer segmentation saved successfully.")