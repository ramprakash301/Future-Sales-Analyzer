
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

query = """
SELECT
    category,
    ROUND(SUM(net_sales), 2) AS total_sales
FROM sales_data
GROUP BY category
ORDER BY total_sales DESC;
"""

df = pd.read_sql(query, connection)

connection.close()

print("Category-wise sales:")
print(df)