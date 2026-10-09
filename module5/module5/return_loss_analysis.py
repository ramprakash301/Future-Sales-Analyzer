import pandas as pd

# Load cleaned dataset
df = pd.read_csv("module1/data/processed/ecommerce_sales_cleaned.csv")

# Select returned orders only
returned_orders = df[df["returned"] == "Yes"].copy()

# Calculate estimated loss only for orders with negative profit margin
returned_orders["estimated_return_loss"] = 0.0

loss_mask = returned_orders["profit_margin"] < 0

returned_orders.loc[loss_mask, "estimated_return_loss"] = (
    returned_orders.loc[loss_mask, "net_sales"]
    * returned_orders.loc[loss_mask, "profit_margin"].abs()
    / 100
)

# Summary calculations
total_returned_orders = len(returned_orders)
loss_returned_orders = loss_mask.sum()
profit_returned_orders = total_returned_orders - loss_returned_orders

total_returned_sales = returned_orders["net_sales"].sum()
estimated_return_loss = returned_orders["estimated_return_loss"].sum()

# Return loss percentage
if total_returned_sales > 0:
    return_loss_percent = (
        estimated_return_loss / total_returned_sales
    ) * 100
else:
    return_loss_percent = 0

# Display results
print("\n===== RETURN LOSS ANALYSIS =====")

print(f"Total Returned Orders       : {total_returned_orders}")
print(f"Returned Orders with Loss   : {loss_returned_orders}")
print(f"Returned Orders with Profit : {profit_returned_orders}")

print(f"Returned Sales Value        : {total_returned_sales:.2f}")
print(f"Estimated Return Loss       : {estimated_return_loss:.2f}")
print(f"Estimated Return Loss %     : {return_loss_percent:.2f}%")

# Category-wise estimated loss
category_loss = (
    returned_orders
    .groupby("category")
    .agg(
        returned_orders=("order_id", "count"),
        returned_sales=("net_sales", "sum"),
        estimated_return_loss=("estimated_return_loss", "sum")
    )
    .reset_index()
)

print("\nCategory-wise Return Loss:")
print(category_loss)

# Save results
returned_orders.to_csv(
    "module5_return_loss_orders.csv",
    index=False
)

category_loss.to_csv(
    "module5_category_return_loss.csv",
    index=False
)

print("\nReturn Loss Analysis completed successfully.")
print("Files saved:")
print("module5_return_loss_orders.csv")
print("module5_category_return_loss.csv")