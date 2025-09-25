# Task 3: Linear Regression – House Price Prediction

This project is **ELEVATE LABS Internship – Task 3**  
It demonstrates how to implement **simple linear regression** to predict house prices using a single feature (`area`) and visualize the regression line.

# Files

- Housing.csv -> Raw dataset  
- LinearRegression.py -> Python script implementing linear regression

# Requirements

Install dependencies:
pip install pandas matplotlib scikit-learn

# Steps Performed

- Load dataset (Housing.csv)
- Handle missing values
- Select feature (area) and target (price)
- Split data into train-test sets
- Fit a Linear Regression model using sklearn.linear_model
- Evaluate the model using MAE, MSE, and R² Score
- Print model intercept and coefficient
- Plot regression line showing predicted vs actual house prices