import pandas as pd

# Load prediction output
df = pd.read_csv("sales_predictions.csv")

# Calculate error
df["Error"] = (
    df["Actual_Net_Sales"] - df["Predicted_Net_Sales"]
)

df["Absolute_Error"] = df["Error"].abs()

# Display summary
print("Prediction Error Analysis")
print("-------------------------")

print("Average Absolute Error:",
      round(df["Absolute_Error"].mean(), 2))

print("Maximum Absolute Error:",
      round(df["Absolute_Error"].max(), 2))

print()
print("Top 10 Prediction Errors:")
print(
    df.sort_values(
        by="Absolute_Error",
        ascending=False
    ).head(10)
)

# Save analysis
df.to_csv(
    "prediction_error_analysis.csv",
    index=False
)

print()
print("Prediction error analysis saved successfully!")
print("File: prediction_error_analysis.csv")