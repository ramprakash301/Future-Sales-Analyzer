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

# Convert order_date to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Create daily sales
daily_sales = (
    df.groupby("order_date")["net_sales"]
    .sum()
    .reset_index()
)

print("Daily sales created successfully!")
print(daily_sales.head())
print("Shape:", daily_sales.shape)