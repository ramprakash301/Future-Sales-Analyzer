import pandas as pd
import matplotlib.pyplot as plt

# Load prediction error data
df = pd.read_csv("prediction_error_analysis.csv")

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Actual_Net_Sales"],
    df["Predicted_Net_Sales"],
    alpha=0.5
)

plt.xlabel("Actual Net Sales")
plt.ylabel("Predicted Net Sales")
plt.title("Actual vs Predicted Net Sales")

plt.tight_layout()
plt.show()