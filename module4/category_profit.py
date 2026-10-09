
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
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_data
GROUP BY category
ORDER BY total_profit DESC;
"""

df = pd.read_sql(query, connection)

connection.close()

print("Category-wise Profit:")
print(df)