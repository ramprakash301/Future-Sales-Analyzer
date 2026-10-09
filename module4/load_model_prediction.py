import joblib
import pandas as pd

# Load saved model
model = joblib.load("lightgbm_sales_model.pkl")

# Sample input data
sample_data = pd.DataFrame({
    "price": [249.99],
    "discount": [10],
    "quantity": [2],
    "delivery_time_days": [5],
    "shipping_cost": [15],
    "customer_age": [25],
    "year": [2025],
    "month": [9],
    "day": [15],
    "day_of_week": [1],
    "week_of_year": [38],
    "discount_percent": [10]
})

# Make prediction
prediction = model.predict(sample_data)

print("Model Loaded Successfully!")
print("Predicted Net Sales:", round(prediction[0], 2))