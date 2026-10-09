
import os

import psycopg2
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# PostgreSQL connection
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

# Features
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

X = df[features]
y = df["net_sales"]

# Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load saved model
model = joblib.load("lightgbm_sales_model.pkl")

# Predictions
predictions = model.predict(X_test)

# Create output dataframe
prediction_df = pd.DataFrame({
    "Actual_Net_Sales": y_test.values,
    "Predicted_Net_Sales": predictions
})

# Save predictions
prediction_df.to_csv(
    "sales_predictions.csv",
    index=False
)

print("Prediction output generated successfully!")
print("Prediction rows:", len(prediction_df))
print("File: sales_predictions.csv")
print()
print(prediction_df.head(10))