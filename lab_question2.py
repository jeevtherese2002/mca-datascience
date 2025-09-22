# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the California Housing dataset
# This is a standard dataset, likely the same as your 'mlr.csv'
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='HousePrice')

# For consistency, let's combine them into one dataframe
df = pd.concat([X, y], axis=1)

print("Dataset loaded successfully!")

#question 1


# Display the first 10 rows of the dataset
print("First 10 rows of the dataset:")
print(df.head(10))

# Compute basic statistics for each feature
print("\nBasic statistics for each feature:")
print(X.describe())

#question 2

# Calculate the correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Housing Features')
plt.show()


#question 3
# 1. Multiple Regression Model (for comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
multi_reg = LinearRegression()
multi_reg.fit(X_train, y_train)
multi_r2 = multi_reg.score(X_test, y_test)
print(f"R² score for Multiple Regression Model: {multi_r2:.4f}")

# 2. Simple Linear Regression for 'MedInc'
X_single = X[['MedInc']] # Feature needs to be a 2D array
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y, test_size=0.2, random_state=42)

simple_reg = LinearRegression()
simple_reg.fit(X_train_s, y_train_s)
simple_r2 = simple_reg.score(X_test_s, y_test_s)
print(f"R² score for Simple Regression Model (MedInc only): {simple_r2:.4f}")

#question 4

# Use the multiple regression model from the previous step
y_pred = multi_reg.predict(X_test)
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

#question 5

# 1. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split and train the model on scaled data
X_train_sc, X_test_sc, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
scaled_model = LinearRegression()
scaled_model.fit(X_train_sc, y_train)
scaled_r2 = scaled_model.score(X_test_sc, y_test)

# 3. Compare R² scores and coefficients
print(f"R² score without scaling: {multi_r2:.4f}")
print(f"R² score with scaling: {scaled_r2:.4f}")

print("\nCoefficients without scaling:\n", multi_reg.coef_)
print("\nCoefficients with scaling:\n", scaled_model.coef_)

#question 6

# Use the scaled data from the previous step
# Linear Regression (already trained)
print(f"Linear Regression R²: {scaled_r2:.4f}")

# Ridge Regression
ridge_model = Ridge(alpha=1.0) # alpha is the regularization strength
ridge_model.fit(X_train_sc, y_train)
ridge_r2 = ridge_model.score(X_test_sc, y_test)
print(f"Ridge Regression R²: {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.01) # A small alpha is often needed for Lasso
lasso_model.fit(X_train_sc, y_train)
lasso_r2 = lasso_model.score(X_test_sc, y_test)
print(f"Lasso Regression R²: {lasso_r2:.4f}")


#question 7

train_sizes = [0.5, 0.7, 0.9]
r2_scores = []

for size in train_sizes:
    # We only need the training set for this experiment
    X_train_exp, _, y_train_exp, _ = train_test_split(X_scaled, y, train_size=size, random_state=42)

    model_exp = LinearRegression()
    model_exp.fit(X_train_exp, y_train_exp)

    # Evaluate on the original, consistent test set
    score = model_exp.score(X_test_sc, y_test)
    r2_scores.append(score)
    print(f"Training size {int(size * 100)}% -> R² score: {score:.4f}")

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, r2_scores, marker='o')
plt.title('Model Performance vs. Training Size')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score on Test Set')
plt.grid(True)
plt.show()

#question 8

# Use the scaled data (X_scaled) and the full target set (y)
model_cv = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42) # 5-Fold CV

# Perform cross-validation
cv_scores = cross_val_score(model_cv, X_scaled, y, cv=kfold, scoring='r2')

print(f"Cross-Validation R² scores: {np.round(cv_scores, 4)}")
print(f"Average R² score: {cv_scores.mean():.4f}")
print(f"Standard Deviation of R² scores: {cv_scores.std():.4f}")

#question 9

# Plot boxplots of key features
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['MedInc'])
plt.title('Boxplot of Median Income')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['AveRooms'])
plt.title('Boxplot of Average Rooms')
plt.tight_layout()
plt.show()

# Remove extreme outliers from 'AveRooms'
q1 = df['AveRooms'].quantile(0.25)
q3 = df['AveRooms'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr

df_no_outliers = df[(df['AveRooms'] > lower_bound) & (df['AveRooms'] < upper_bound)]

print(f"Original dataset size: {len(df)}")
print(f"Dataset size after removing outliers: {len(df_no_outliers)}")

# Retrain the model on the cleaned data
X_clean = df_no_outliers.drop('HousePrice', axis=1)
y_clean = df_no_outliers['HousePrice']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
model_clean = LinearRegression()
model_clean.fit(X_train_c, y_train_c)
clean_r2 = model_clean.score(X_test_c, y_test_c)

print(f"\nR² score with original data: {multi_r2:.4f}")
print(f"R² score after removing outliers: {clean_r2:.4f}")

#question 10

# Create a scatter plot of location vs. house prices
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Longitude',
    y='Latitude',
    data=df,
    hue='HousePrice',
    palette='viridis',
    s=20, # size of points
    alpha=0.6
)
plt.title('California Housing Prices by Geographic Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='House Price (in $100k)')
plt.show()


