import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from GetData import get_data

real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data()

top_n_variables = len(real_time_X.columns) // 3

# Train real time Random Forest model
real_time_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
real_time_rf_model.fit(real_time_X, real_time_y.values.ravel())

# Get feature importance scores
real_time_feature_importance = real_time_rf_model.feature_importances_

# Create a DataFrame to store feature importance scores
real_time_feature_importance_df = pd.DataFrame({'Feature': real_time_X.columns, 'Importance': real_time_feature_importance})
real_time_feature_importance_df = real_time_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print("Real Time Feature Importance Scores:")
print(real_time_feature_importance_df)

# Choose the top N variables based on feature importance
real_time_selected_variables = real_time_feature_importance_df.head(top_n_variables)['Feature'].tolist()
real_time_selected_variables_to_latest = []
for var in real_time_selected_variables:
    real_time_selected_variables_to_latest.append(var[:-4] + curr_year[-2:] + "Q" + curr_quarter)
print("\nTop", top_n_variables, "Real Time Variables Selected:", real_time_selected_variables)

# Train a new random forest model using only the selected variables
real_time_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
print(latest_X_train)
print(real_time_selected_variables_to_latest)
real_time_rf_model_selected.fit(latest_X_train[real_time_selected_variables_to_latest], latest_y_train.values.ravel())

# Evaluate the model's performance
real_time_y_pred = real_time_rf_model_selected.predict(latest_X_test[real_time_selected_variables_to_latest])
real_time_rmsfe = mean_squared_error(latest_y_test, real_time_y_pred)**(0.5)
print("\nRoot Mean Squared Forecast Error (RMSFE) with Real Time Selected Variables:", real_time_rmsfe)

# Train latest Random Forest model
latest_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
latest_rf_model.fit(latest_X_train, latest_y_train.values.ravel())

# Get feature importance scores
latest_feature_importance = latest_rf_model.feature_importances_

# Create a DataFrame to store feature importance scores
latest_feature_importance_df = pd.DataFrame({'Feature': latest_X_train.columns, 'Importance': latest_feature_importance})
latest_feature_importance_df = latest_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print("Latest Feature Importance Scores:")
print(latest_feature_importance_df)

# Choose the top N variables based on feature importance
latest_selected_variables = latest_feature_importance_df.head(top_n_variables)['Feature'].tolist()
latest_selected_variables_to_latest = []
for var in latest_selected_variables:
    latest_selected_variables_to_latest.append(var[:-4] + curr_year[-2:] + "Q" + curr_quarter)
print("\nTop", top_n_variables, "Latest Variables Selected:", latest_selected_variables)

# Train a new random forest model using only the selected variables, using latest vintage data
latest_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
latest_rf_model_selected.fit(latest_X_train[latest_selected_variables_to_latest], latest_y_train.values.ravel())

# Evaluate the model's performance
latest_y_pred = latest_rf_model_selected.predict(latest_X_test[latest_selected_variables_to_latest])
latest_rmsfe = mean_squared_error(latest_y_test, latest_y_pred)**(0.5)
print("\nRoot Mean Squared Forecast Error (RMSFE) with Latest Selected Variables:", latest_rmsfe)