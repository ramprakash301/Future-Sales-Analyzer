import os

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="ecommerce_sales_db",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

query = """
SELECT
    product_id,
    ROUND(SUM(net_sales), 2) AS total_sales
FROM sales_data
GROUP BY product_id
ORDER BY total_sales DESC
LIMIT 10;
"""

df = pd.read_sql(query, connection)

connection.close()

# Top 10 products graph
plt.figure(figsize=(10, 6))
plt.bar(df["product_id"], df["total_sales"])

plt.title("Top 10 Products by Sales")
plt.xlabel("Product ID")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()