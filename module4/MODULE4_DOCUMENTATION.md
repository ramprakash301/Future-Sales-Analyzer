# E-Commerce Sales Prediction - Module 4

## Objective

The objective of this module is to analyze e-commerce sales data
and build a machine learning model to predict net sales.

## Dataset

Dataset Shape:
34500 rows and 27 columns

Missing Values:
0

Duplicate Rows:
0

## Data Analysis

The following analysis was performed:

- Daily Sales Analysis
- Monthly Sales Analysis
- Category Sales Analysis
- Category Profit Analysis
- Product Sales Analysis
- Monthly Sales Extremes
- Sales Visualizations

## Feature Selection

Selected Features:

- price
- discount
- quantity
- delivery_time_days
- shipping_cost
- customer_age
- year
- month
- day
- day_of_week
- week_of_year
- discount_percent

Target Variable:

net_sales

## Train-Test Split

Training Data:
27600 rows

Testing Data:
6900 rows

## Machine Learning Model

Model Used:

LightGBM

The LightGBM model was trained using the selected features
to predict net sales.

## Model Evaluation

MAE  : 4.76

RMSE : 39.72

R2 Score : 0.9867

## Feature Importance

Top Important Features:

1. price
2. quantity
3. shipping_cost
4. discount
5. day

## Prediction

The trained model successfully generated predictions
for 6900 test records.

Prediction file:

sales_predictions.csv

## Model Saving

The trained model was saved as:

lightgbm_sales_model.pkl

## Error Analysis

Prediction errors were analyzed using:

prediction_error_analysis.csv

Maximum Absolute Error:

2529.29

## Conclusion

The LightGBM model successfully predicts e-commerce net sales
with an R2 score of 0.9867.

The project demonstrates data analysis, visualization,
feature selection, machine learning model training,
prediction, evaluation, feature importance analysis,
and model saving.