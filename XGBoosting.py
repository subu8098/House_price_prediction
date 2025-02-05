from xgboost import XGBRegressor

# Train an XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
train_mse_xgb = mean_squared_error(y_train, y_train_pred_xgb)
test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)

train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

print("XGBoost Training MSE:", train_mse_xgb)
print("XGBoost Testing MSE:", test_mse_xgb)
print("XGBoost Training R² Score:", train_r2_xgb)
print("XGBoost Testing R² Score:", test_r2_xgb)

# Predict on new data
predicted_price_xgb = xgb_model.predict(new_house_df)
print("XGBoost Predicted House Price:", predicted_price_xgb[0] * 1000)