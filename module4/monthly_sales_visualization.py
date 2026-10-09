import os

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Connect to PostgreSQL
connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="ecommerce_sales_db",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

# Fetch data
query = "SELECT * FROM sales_data;"
df = pd.read_sql(query, connection)

connection.close()

# Convert date
df["order_date"] = pd.to_datetime(df["order_date"])

# Create month
df["month"] = df["order_date"].dt.to_period("M")

# Calculate monthly sales
monthly_sales = (
    df.groupby("month")["net_sales"]
    .sum()
    .reset_index()
)

# Convert period to string for plotting
monthly_sales["month"] = monthly_sales["month"].astype(str)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["month"], monthly_sales["net_sales"], marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Net Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()