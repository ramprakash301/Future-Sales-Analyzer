
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
    category,
    ROUND(SUM(net_sales), 2) AS total_sales
FROM sales_data
GROUP BY category
ORDER BY total_sales DESC;
"""

df = pd.read_sql(query, connection)

connection.close()

# Category-wise sales graph
plt.figure(figsize=(10, 6))
plt.bar(df["category"], df["total_sales"])

plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()