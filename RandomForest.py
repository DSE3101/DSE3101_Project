import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the macro variables
with open('preprocessed_data.pkl', 'rb') as f:
    (RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT) = pickle.load(f)

macro_variables = [RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT]

# Create YYQq to slice vintages by index
vintages = ["65Q4"]
vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
vintages.extend([f'{i}Q{j}' for i in range(10, 24) for j in range(1, 5)])
vintages.extend(["24Q1"])

# year_input = input("Choose real time data from 1966 to 2023")
year_input = "2012"
# quarter_input = input("Choose a quarter from 1 to 4")
quarter_input = "2"

# Combine all variables for chosen quarter and remove rows
def get_vintage_data(year, quarter):
    quarter_index = vintages.index(f'{year[-2:]}Q{quarter}')
    df = []
    for var in macro_variables:
        df.append(var.iloc[:, quarter_index])
    df = pd.concat(df, axis=1)
    # Remove rows after chosen quarter
    df = df[df.index <= f"{year}:Q{quarter}"]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

real_time_X, real_time_y = get_vintage_data(year_input, quarter_input)
latest_X, latest_y = get_vintage_data(ROUTPUT.columns[-1][-4:-2], ROUTPUT.columns[-1][-1])





# Train Test Split
latest_X_train, latest_X_test, latest_y_train, latest_y_test = train_test_split(latest_X, latest_y, test_size=0.2, random_state=42)

# Train a random forest model
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
top_n_variables = 21  # You can adjust this value based on your preference
latest_selected_variables = latest_feature_importance_df.head(top_n_variables)['Feature'].tolist()
print("\nTop", top_n_variables, "Latest Variables Selected:", latest_selected_variables)

# Train a new random forest model using only the selected variables
latest_X_train_selected = latest_X_train[latest_selected_variables]
latest_X_test_selected = latest_X_test[latest_selected_variables]

latest_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
latest_rf_model_selected.fit(latest_X_train_selected, latest_y_train)

# Evaluate the model's performance
latest_y_pred = latest_rf_model_selected.predict(latest_X_test_selected)
latest_mse = mean_squared_error(latest_y_test, latest_y_pred)
print("\nMean Squared Error (MSE) with Latest Selected Variables:", latest_mse)


# Train Test Split
real_time_X_train, real_time_X_test, real_time_y_train, real_time_y_test = train_test_split(real_time_X, real_time_y, test_size=0.2, random_state=42)

# Train a random forest model
real_time_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
real_time_rf_model.fit(real_time_X_train, real_time_y_train)

# Get feature importance scores
real_time_feature_importance = real_time_rf_model.feature_importances_

# Create a DataFrame to store feature importance scores
real_time_feature_importance_df = pd.DataFrame({'Feature': real_time_X.columns, 'Importance': real_time_feature_importance})
real_time_feature_importance_df = real_time_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print("Real Time Feature Importance Scores:")
print(real_time_feature_importance_df)

# Choose the top N variables based on feature importance
top_n_variables = 21  # You can adjust this value based on your preference
real_time_selected_variables = real_time_feature_importance_df.head(top_n_variables)['Feature'].tolist()
print("\nTop", top_n_variables, "Real Time Variables Selected:", real_time_selected_variables)

# Train a new random forest model using only the selected variables
real_time_X_train_selected = real_time_X_train[real_time_selected_variables]
real_time_X_test_selected = real_time_X_test[real_time_selected_variables]

real_time_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
real_time_rf_model_selected.fit(real_time_X_train_selected, real_time_y_train)

# Evaluate the model's performance
real_time_y_pred = real_time_rf_model_selected.predict(real_time_X_test_selected)
real_time_mse = mean_squared_error(real_time_y_test, real_time_y_pred)
print("\nMean Squared Error (MSE) with Real Time Selected Variables:", real_time_mse)