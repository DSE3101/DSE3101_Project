import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the macro variables
with open('preprocessed_data.pkl', 'rb') as f:
    macro_variables = pickle.load(f)

# Combine all variables for chosen quarter and remove rows
def get_vintage_data(vintage_year, vintage_quarter, data_year, data_quarter):
    quarter_index = vintages.index(f'{vintage_year[-2:]}Q{vintage_quarter}')
    df = []
    for var in macro_variables:
        df.append(var.iloc[:, quarter_index])
    df = pd.concat(df, axis=1)
    # Remove rows after chosen quarter
    df = df[df.index <= f"{data_year}:Q{data_quarter}"]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# Get input
# chosen_variable = macro_variables[macro_variables.index(input("Choose a macro variable to forecast"))]
chosen_variable = macro_variables[-1]
curr_year = chosen_variable.columns[-1][-4:-2]
curr_quarter = chosen_variable.columns[-1][-1]
# year_input = input("Choose real time data from 1966 to 2023")
year_input = "2012"
# quarter_input = input("Choose a quarter from 1 to 4")
quarter_input = "2"
# h_step_input = input("Choose number of steps to forecast")
h_step_input = 11
years_ahead = h_step_input // 4
quarters_ahead = h_step_input % 8
h_step_year = int(year_input) + years_ahead
h_step_quarter = int(quarter_input) + quarters_ahead
if h_step_quarter > 4:
    h_step_year += 1
    h_step_quarter -= 4
h_step_year = str(h_step_year)
h_step_quarter = str(h_step_quarter)

# Create YYQq to slice vintages by index
vintages = ["65Q4"]
vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
vintages.extend([f'{i}Q{j}' for i in range(10, int(curr_year)) for j in range(1, 5)])
vintages.extend([f'{int(curr_year)}Q{j}' for j in range(1, int(curr_quarter)+1)])

# Slice data as needed
real_time_X, real_time_y = get_vintage_data(year_input, quarter_input, year_input, quarter_input)
latest_X, latest_y = get_vintage_data(curr_year, curr_quarter, h_step_year, h_step_quarter)
latest_X_train = latest_X.iloc[:len(real_time_y), :]
latest_y_train = latest_y.iloc[:len(real_time_y)]
latest_X_test = latest_X.iloc[len(real_time_y):, :]
latest_y_test = latest_y.iloc[len(real_time_y):]
top_n_variables = len(macro_variables) // 3

# Train real time Random Forest model
real_time_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
real_time_rf_model.fit(real_time_X, real_time_y)

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
    real_time_selected_variables_to_latest.append(var[:-4] + curr_year + "Q" + curr_quarter)
print("\nTop", top_n_variables, "Latest Variables Selected:", real_time_selected_variables)

# Train a new random forest model using only the selected variables
real_time_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
real_time_rf_model_selected.fit(latest_X_train[real_time_selected_variables_to_latest], latest_y_train)

# Evaluate the model's performance
real_time_y_pred = real_time_rf_model_selected.predict(latest_X_test[real_time_selected_variables_to_latest])
real_time_rmsfe = mean_squared_error(latest_y_test, real_time_y_pred)**(0.5)
print("\nRoot Mean Squared Forecast Error (RMSFE) with Real Time Selected Variables:", real_time_rmsfe)

# Train latest Random Forest model
latest_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
latest_rf_model.fit(latest_X_train, latest_y_train)

# Get feature importance scores
latest_feature_importance = latest_rf_model.feature_importances_

# Create a DataFrame to store feature importance scores
latest_feature_importance_df = pd.DataFrame({'Feature': latest_X.columns, 'Importance': latest_feature_importance})
latest_feature_importance_df = latest_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print("Latest Feature Importance Scores:")
print(latest_feature_importance_df)

# Choose the top N variables based on feature importance
latest_selected_variables = latest_feature_importance_df.head(top_n_variables)['Feature'].tolist()
latest_selected_variables_to_latest = []
for var in latest_selected_variables:
    latest_selected_variables_to_latest.append(var[:-4] + curr_year + "Q" + curr_quarter)
print("\nTop", top_n_variables, "Latest Variables Selected:", latest_selected_variables)

# Train a new random forest model using only the selected variables, using latest vintage data
latest_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
latest_rf_model_selected.fit(latest_X_train[latest_selected_variables_to_latest], latest_y_train)

# Evaluate the model's performance
latest_y_pred = latest_rf_model_selected.predict(latest_X_test[latest_selected_variables_to_latest])
latest_rmsfe = mean_squared_error(latest_y_test, latest_y_pred)**(0.5)
print("\nRoot Mean Squared Forecast Error (RMSFE) with Latest Selected Variables:", latest_rmsfe)