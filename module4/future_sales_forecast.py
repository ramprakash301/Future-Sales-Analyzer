import pandas as pd

# Load processed dataset
df = pd.read_csv("module1/data/processed/ecommerce_sales_cleaned.csv")

# Convert order_date to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Month-wise sales
monthly_sales = (
    df.groupby(df["order_date"].dt.to_period("M"))["net_sales"]
    .sum()
    .reset_index()
)

# Convert period to date
monthly_sales["order_date"] = monthly_sales["order_date"].dt.to_timestamp()

print("\nPast Monthly Sales:")
print(monthly_sales)

# Save historical sales
monthly_sales.to_csv(
    "module4/monthly_historical_sales.csv",
    index=False
)

print("\nHistorical monthly sales saved successfully.")


# Create time-based features
monthly_sales["year"] = monthly_sales["order_date"].dt.year
monthly_sales["month"] = monthly_sales["order_date"].dt.month

# Create a sequential time index
monthly_sales["time_index"] = range(len(monthly_sales))

print("\nMonthly Sales with Time Features:")
print(monthly_sales)



# Create lag features for time-series forecasting

# Remove incomplete September 2025 data
forecast_data = monthly_sales[
    monthly_sales["order_date"] < "2025-09-01"
].copy()

# Previous month sales
forecast_data["lag_1"] = forecast_data["net_sales"].shift(1)

# Sales from 2 months before
forecast_data["lag_2"] = forecast_data["net_sales"].shift(2)

# Sales from 3 months before
forecast_data["lag_3"] = forecast_data["net_sales"].shift(3)

# Average sales of previous 3 months
forecast_data["rolling_3"] = (
    forecast_data["net_sales"]
    .shift(1)
    .rolling(3)
    .mean()
)

# Remove rows where lag values are not available
forecast_data = forecast_data.dropna().reset_index(drop=True)

print("\nForecasting Dataset:")
print(forecast_data)

print("\nForecasting Dataset Shape:")
print(forecast_data.shape)



# Step 4 - Train and test the forecasting model

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Features used by the model
features = [
    "year",
    "month",
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_3"
]

target = "net_sales"

# Use the last 6 months for testing
train_data = forecast_data.iloc[:-6]
test_data = forecast_data.iloc[-6:]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Create LightGBM model
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    min_child_samples=3,
    random_state=42,
    verbosity=-1
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
test_predictions = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, test_predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

print("\nForecasting Model Evaluation")
print("--------------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

print("\nActual vs Predicted Sales:")
comparison = pd.DataFrame({
    "order_date": test_data["order_date"],
    "actual_sales": y_test.values,
    "predicted_sales": test_predictions
})

print(comparison)




# Step 5 - Forecast Future Monthly Sales

# Train the final model using all available historical data
X_full = forecast_data[features]
y_full = forecast_data[target]

final_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    min_child_samples=3,
    random_state=42,
    verbosity=-1
)

final_model.fit(X_full, y_full)
import joblib

joblib.dump(
    final_model,
    "module4/future_sales_model.pkl"
)

print("Future sales model saved successfully!")

# Store historical sales for recursive forecasting
history = forecast_data[
    ["order_date", "net_sales"]
].copy()

future_predictions = []

# Generate next 6 months
for i in range(6):

    next_date = (
        history["order_date"].max()
        + pd.DateOffset(months=1)
    )

    # Get previous sales values
    lag_1 = history["net_sales"].iloc[-1]
    lag_2 = history["net_sales"].iloc[-2]
    lag_3 = history["net_sales"].iloc[-3]

    rolling_3 = (
        history["net_sales"]
        .iloc[-3:]
        .mean()
    )

    # Create future input
    future_input = pd.DataFrame({
        "year": [next_date.year],
        "month": [next_date.month],
        "time_index": [len(monthly_sales) + i],
        "lag_1": [lag_1],
        "lag_2": [lag_2],
        "lag_3": [lag_3],
        "rolling_3": [rolling_3]
    })

    # Predict future sales
    prediction = final_model.predict(
        future_input[features]
    )[0]

    # Save prediction
    future_predictions.append({
        "order_date": next_date,
        "predicted_net_sales": prediction
    })

    # Add prediction to history
    history = pd.concat([
        history,
        pd.DataFrame({
            "order_date": [next_date],
            "net_sales": [prediction]
        })
    ], ignore_index=True)


# Convert predictions to DataFrame
future_forecast = pd.DataFrame(future_predictions)

print("\nFuture Sales Forecast")
print("----------------------")
print(future_forecast)

# Save forecast results
future_forecast.to_csv(
    "module4/future_sales_predictions.csv",
    index=False
)

print("\nFuture sales forecast saved successfully.")



# Step 7 - Combine Historical Sales and Future Forecast

# Historical completed months only
historical_data = monthly_sales[
    monthly_sales["order_date"] < "2025-09-01"
][["order_date", "net_sales"]].copy()

historical_data["sales_type"] = "Historical"

# Rename forecast column
forecast_data_powerbi = future_forecast.rename(
    columns={"predicted_net_sales": "net_sales"}
).copy()

forecast_data_powerbi["sales_type"] = "Forecast"

# Keep only required columns
forecast_data_powerbi = forecast_data_powerbi[
    ["order_date", "net_sales", "sales_type"]
]

# Combine historical and forecast data
combined_sales = pd.concat(
    [historical_data, forecast_data_powerbi],
    ignore_index=True
)

# Save combined data
combined_sales.to_csv(
    "module4/historical_and_forecast_sales.csv",
    index=False
)

print("\nHistorical + Forecast data:")
print(combined_sales)

print("\nCombined sales data saved successfully.")