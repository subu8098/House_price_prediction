# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from local CSV
file_path = r"C:\Users\SUBRAMANI V\Downloads\HousingData.csv"
data = pd.read_csv(file_path)

# Explore the data
print(data.head())
print(data.describe())
print(data.isnull().sum())

# Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Separate features (X) and target (y)
X = data.drop('MEDV', axis=1)  # Features
y = data['MEDV']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Linear Regression
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)

train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print("Linear Regression Training MSE:", train_mse_lr)
print("Linear Regression Testing MSE:", test_mse_lr)
print("Linear Regression Training R² Score:", train_r2_lr)
print("Linear Regression Testing R² Score:", test_r2_lr)

# Evaluate Random Forest
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)

train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

print("Random Forest Training MSE:", train_mse_rf)
print("Random Forest Testing MSE:", test_mse_rf)
print("Random Forest Training R² Score:", train_r2_rf)
print("Random Forest Testing R² Score:", test_r2_rf)

# Make predictions on new data
new_house = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]])

# Convert to a DataFrame with feature names
new_house_df = pd.DataFrame(new_house, columns=X.columns)

# Predict the price
predicted_price = rf_model.predict(new_house_df)
print("Predicted House Price:", predicted_price[0] * 1000)