import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import & preprocess dataset
df = pd.read_csv("Housing.csv")
df = df.dropna()

# Use only one feature to visualize regression line
X = df[['area']]
y = df['price']

# 2. Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("Evaluations:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient for area:", model.coef_[0])

# 5. Plot regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction - Linear Regression")
plt.legend()
plt.show()