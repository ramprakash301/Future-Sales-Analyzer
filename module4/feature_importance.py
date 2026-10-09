import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
importance.to_csv("module4/feature_importance.csv", index=False)
print("Feature Importance:")
print(importance)

# Visualization
plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("LightGBM Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()