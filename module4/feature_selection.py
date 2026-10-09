
import os

import psycopg2
import pandas as pd

connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="ecommerce_sales_db",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

query = "SELECT * FROM sales_data;"

df = pd.read_sql(query, connection)

connection.close()

# Features selected for sales forecasting
features = [
    "price",
    "discount",
    "quantity",
    "delivery_time_days",
    "shipping_cost",
    "customer_age",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "discount_percent"
]

target = "net_sales"

X = df[features]
y = df[target]

print("Selected Features:")
print(features)

print("\nTarget Variable:")
print(target)

print("\nFeature Data Shape:", X.shape)
print("Target Data Shape:", y.shape)