import os
import psycopg2

try:
    connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="ecommerce_sales_db",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

    print("PostgreSQL connected successfully!")

    connection.close()

except Exception as e:
    print("Connection failed!")
    print(e)