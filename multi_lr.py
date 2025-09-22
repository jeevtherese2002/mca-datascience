"""
Program: Multiple Linear Regression using California Housing Dataset
-Dataset Explanation:
- The California Housing dataset is included in scikit-learn.
- It contains information from the 1990 U.S. Census about California districts.
- Features include:
MedInc   = Median income in block group
HouseAge = Median house age
AveRooms = Average number of rooms
AveBedrms= Average number of bedrooms
Population = Block group population
AveOccup = Average house occupancy
Latitude, Longitude = Location
- Target variable: Median house value (in 100,000s USD).
- Goal: Predict house prices using multiple features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Regression equation (coefficients for each feature)
print(" Intercept (b0) : ", model.intercept_)
print(" Coefficients (b1, b2, ..., bn) : ")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Evaluation
print("\n Mean Squared Error : ", mean_squared_error(y_test, y_pred))
print(" R2 Score : ", r2_score(y_test, y_pred))

# Visualization: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="green", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],color="red",linestyle='--')
plt.xlabel(" Actual Prices (in $100,000s) ")
plt.ylabel(" Predicted Prices (in $100,000s) ")
plt.title(" Multiple Linear Regression - California Housing Dataset ")
plt.grid(True)
plt.show()