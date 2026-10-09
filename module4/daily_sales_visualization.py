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

# Create daily sales
daily_sales = (
    df.groupby("order_date")["net_sales"]
    .sum()
    .reset_index()
)

# Plot daily sales trend
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["order_date"], daily_sales["net_sales"])

plt.title("Daily Sales Trend")
plt.xlabel("Order Date")
plt.ylabel("Net Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()