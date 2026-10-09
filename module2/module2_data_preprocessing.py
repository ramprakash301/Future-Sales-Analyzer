import pandas as pd

# ==========================================
# MODULE 2 - DATA PREPROCESSING
# ==========================================

# Load raw dataset
df = pd.read_csv(
    "module1/data/raw/ecommerce_sales_34500.csv"
)

print("Dataset loaded successfully.")
print("Dataset Shape:", df.shape)




# ==========================================
# DATE CONVERSION
# ==========================================

df["order_date"] = pd.to_datetime(df["order_date"])

print("\nOrder Date Data Type:")
print(df["order_date"].dtype)




# ==========================================
# DATE FEATURE ENGINEERING
# ==========================================

df["year"] = df["order_date"].dt.year

df["month"] = df["order_date"].dt.month

df["day"] = df["order_date"].dt.day

df["day_of_week"] = df["order_date"].dt.dayofweek

df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)

print("\nDate Features:")
print(
    df[
        [
            "order_date",
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year"
        ]
    ].head()
)

df["returned_flag"] = df["returned"].map({
    "Yes": 1,
    "No": 0
})

print("\nReturn Status:")
print(df[["returned", "returned_flag"]].head())

df["discount_percent"] = df["discount"] * 100

print("\nDiscount Percentage:")
print(df[["discount", "discount_percent"]].head())

df["gross_sales"] = df["price"] * df["quantity"]

print("\nGross Sales:")
print(df[["price", "quantity", "gross_sales"]].head())

df["net_sales"] = df["price"] * df["quantity"] * (1 - df["discount"])

print("\nNet Sales:")
print(df[["price", "quantity", "discount", "net_sales"]].head())

# ==========================================
# PROFIT / LOSS STATUS
# ==========================================

df["profit_loss_status"] = df["profit_margin"].apply(
    lambda x: "Profit" if x > 0 else "Loss"
)

print("\nProfit / Loss Status:")
print(
    df[
        ["profit_margin", "profit_loss_status"]
    ].head()
)

# ==========================================
# DATA TYPE CHECK
# ==========================================

print("\nData Types After Preprocessing:")
print(df.dtypes)

# ==========================================
# FINAL MISSING VALUE CHECK
# ==========================================

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# ==========================================
# DUPLICATE CHECK AFTER PREPROCESSING
# ==========================================

print("\nDuplicate Rows After Preprocessing:")
print(df.duplicated().sum())

# ==========================================
# SAVE PROCESSED DATASET
# ==========================================

df.to_csv(
    "module1/data/processed/ecommerce_sales_cleaned.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")