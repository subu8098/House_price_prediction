# Train a Linear Regression model (Multiple Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluate the model
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)

train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print("Linear Regression Training MSE:", train_mse_lr)
print("Linear Regression Testing MSE:", test_mse_lr)
print("Linear Regression Training R² Score:", train_r2_lr)
print("Linear Regression Testing R² Score:", test_r2_lr)

# Predict on new data
predicted_price_lr = lr_model.predict(new_house_df)
print("Linear Regression Predicted House Price:", predicted_price_lr[0] * 1000)