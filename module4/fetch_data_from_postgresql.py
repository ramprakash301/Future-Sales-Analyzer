
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

print("Data fetched successfully!")
print(df.head())
print("Shape:", df.shape)