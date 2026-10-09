import pandas as pd

# Model evaluation results
results = {
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Value": [4.76, 39.72, 0.9867]
}

df = pd.DataFrame(results)

print("Model Performance Summary")
print("-------------------------")
print(df)

# Save summary
df.to_csv("model_performance_summary.csv", index=False)

print()
print("Performance summary saved successfully!")
print("File: model_performance_summary.csv")