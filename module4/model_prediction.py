
import os

import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

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

target = "net_sales"

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Show actual vs predicted values
results = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred
})

print("Predictions Generated Successfully!")
print(results.head(10))

print("\nPrediction Shape:", results.shape)