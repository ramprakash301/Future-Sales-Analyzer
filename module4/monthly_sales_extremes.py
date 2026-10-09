import os

import psycopg2
import pandas as pd

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

# Create monthly sales
df["month"] = df["order_date"].dt.to_period("M")

monthly_sales = (
    df.groupby("month")["net_sales"]
    .sum()
    .reset_index()
)

# Highest and lowest sales months
highest_month = monthly_sales.loc[monthly_sales["net_sales"].idxmax()]
lowest_month = monthly_sales.loc[monthly_sales["net_sales"].idxmin()]

print("Highest Sales Month:")
print(highest_month)

print("\nLowest Sales Month:")
print(lowest_month)