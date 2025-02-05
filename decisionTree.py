from sklearn.tree import DecisionTreeRegressor

# Train a Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

# Evaluate the model
train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)

train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)

print("Decision Tree Training MSE:", train_mse_dt)
print("Decision Tree Testing MSE:", test_mse_dt)
print("Decision Tree Training R² Score:", train_r2_dt)
print("Decision Tree Testing R² Score:", test_r2_dt)

# Predict on new data
predicted_price_dt = dt_model.predict(new_house_df)
print("Decision Tree Predicted House Price:", predicted_price_dt[0] * 1000)