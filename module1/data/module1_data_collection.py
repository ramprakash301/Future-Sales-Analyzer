import pandas as pd

# Load the dataset
df = pd.read_csv("module1/data/raw/ecommerce_sales_34500.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display dataset information
print("\nDataset Information:")
df.info()


# ==============================
# DATA QUALITY CHECK
# ==============================

print("\n--- DATA QUALITY CHECK ---")

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Number of unique customers
print("\nUnique Customers:")
print(df["customer_id"].nunique())

# Number of unique products
print("\nUnique Products:")
print(df["product_id"].nunique())

# Product categories
print("\nCategories:")
print(df["category"].unique())

# Return values
print("\nReturn Values:")
print(df["returned"].unique())

# Date range
print("\nDate Range:")
print("Start Date:", df["order_date"].min())
print("End Date:", df["order_date"].max())


# ==============================
# VALUE RANGE CHECK
# ==============================

print("\n--- VALUE RANGE CHECK ---")

# Numeric summary
print("\nNumeric Summary:")
print(df.describe())

# Negative quantity
print("\nNegative Quantity:")
print((df["quantity"] < 0).sum())

# Negative price
print("\nNegative Price:")
print((df["price"] < 0).sum())

# Negative shipping cost
print("\nNegative Shipping Cost:")
print((df["shipping_cost"] < 0).sum())

# Discount range
print("\nDiscount Range:")
print("Minimum:", df["discount"].min())
print("Maximum:", df["discount"].max())

# Customer age range
print("\nCustomer Age Range:")
print("Minimum:", df["customer_age"].min())
print("Maximum:", df["customer_age"].max())

# Delivery time range
print("\nDelivery Time Range:")
print("Minimum:", df["delivery_time_days"].min())
print("Maximum:", df["delivery_time_days"].max())

