import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "module1/data/processed/ecommerce_sales_cleaned.csv"
)

print("Dataset loaded successfully.")

# Create Profit/Loss Status
df["profit_loss_status"] = df["profit_margin"].apply(
    lambda x: "Profit" if x > 0 else "Loss"
)

# Overall Profit/Loss analysis
profit_orders = (
    df["profit_loss_status"] == "Profit"
).sum()

loss_orders = (
    df["profit_loss_status"] == "Loss"
).sum()

total_orders = len(df)

profit_rate = (profit_orders / total_orders) * 100
loss_rate = (loss_orders / total_orders) * 100

print("\nProfit & Loss Analysis")
print("----------------------")
print("Total Orders:", total_orders)
print("Profit Orders:", profit_orders)
print("Loss Orders:", loss_orders)
print("Profit Order Rate:", round(profit_rate, 2), "%")
print("Loss Order Rate:", round(loss_rate, 2), "%")

# Category-wise Profit/Loss analysis
category_profit_loss = (
    df.groupby("category")
    .agg(
        total_sales=("net_sales", "sum"),
        average_profit_margin=("profit_margin", "mean"),
        profit_orders=("profit_loss_status",
                       lambda x: (x == "Profit").sum()),
        loss_orders=("profit_loss_status",
                     lambda x: (x == "Loss").sum())
    )
    .reset_index()
)

print("\nCategory-wise Profit & Loss:")
print(category_profit_loss)

# Product-wise Profit/Loss analysis
product_profit_loss = (
    df.groupby("product_id")
    .agg(
        total_sales=("net_sales", "sum"),
        average_profit_margin=("profit_margin", "mean"),
        orders=("order_id", "count")
    )
    .reset_index()
    .sort_values(
        "average_profit_margin",
        ascending=False
    )
)

print("\nTop Products by Profit Margin:")
print(product_profit_loss.head(10))

# Save outputs
category_profit_loss.to_csv(
    "module5_category_profit_loss.csv",
    index=False
)

product_profit_loss.to_csv(
    "module5_product_profit_loss.csv",
    index=False
)

print("\nProfit & Loss analysis saved successfully.")