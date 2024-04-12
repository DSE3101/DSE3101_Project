import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from GetData import get_data
from PlotGraphs import *

def ADL_MODEL(year_input, quarter_input):

    # Set the maximum number of lags to test for each variable
    max_lag = 8

    # Initialize LeaveOneOut cross-validator
    loo = LeaveOneOut()

    # Helper function to calculate LOOCV MSE for a single variable at a given lag
    def loocv_mse_for_variable(X, y):
        mse_values = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_values.append(mse)
        return np.mean(mse_values)

    # Get data
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
    candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']
    temp = [f'{var}{year_input[-2:]}Q{quarter_input}' for var in candidate_vars]
    real_time_X = real_time_X.loc[:, temp]

    # Store the optimal lag for each variable
    optimal_lags = {}
    # Iterate over each variable to find its optimal number of lags
    for var in real_time_X.columns:
        best_mse = np.inf
        best_lag = 0
        for lag in range(1, max_lag + 1):
            # Lag the variable
            lagged_series = real_time_X[var].shift(lag)      
            
            # Combine with other variables (without lagging them)
            combined_X = real_time_X.assign(lagged_var=lagged_series).drop(columns=[var]).dropna()
            current_real_time_y = real_time_y.reindex(combined_X.index)
            
            # Calculate LOOCV MSE for the current lag
            mse = loocv_mse_for_variable(combined_X, current_real_time_y)
            
            if mse < best_mse:
                best_mse = mse
                best_lag = lag
        
        # Store the best lag for the variable
        optimal_lags[var] = best_lag
    print(optimal_lags)

    # Loop over each lag to build separate models and perform forecasts
    y_pred = []
    rmsfe = []
    for lag_X in range(1, 9):
        # Initialize lagged features DataFrame
        X_lagged_valid = pd.DataFrame(index=real_time_X.index)

        # Apply optimal lags for each variable
        for var, optimal_lag in optimal_lags.items():
            # Lag the variable by its optimal lag
            lagged_series = real_time_X[var].shift(optimal_lag)
            
            # Append lagged variable to lagged features DataFrame
            X_lagged_valid[f'{var}_lag_{optimal_lag}'] = lagged_series

        # Lag all X variables by the additional lag_X
        X_lagged_valid = X_lagged_valid.shift(lag_X)

        # Drop rows with NaN resulting from lagging
        X_lagged_valid.dropna(inplace=True)

        # Prepare lagged target variable (Yt)
        y_lagged_valid = real_time_y.shift(lag_X).dropna()

        # Fit the model using OLS
        model_lagged = sm.OLS(y_lagged_valid.iloc[len(y_lagged_valid)-len(X_lagged_valid):], X_lagged_valid).fit()

        # Prepare the most recent data point for forecasting Yt
        X_recent = X_lagged_valid.iloc[-1].to_frame().T  # Get the row corresponding to t-1

        # Perform the forecast
        y_pred.append(model_lagged.predict(X_recent)[0])

        # Backtesting
        backtesting_predictions = []
        # print(X_lagged_valid, y_lagged_valid)
        for row in range(len(X_lagged_valid)):
            X_row = X_lagged_valid.iloc[row].to_frame().T
            backtesting_predictions.append(model_lagged.predict(X_row)[0])
        # print(latest_y_train.iloc[len(y_lagged_valid)-len(X_lagged_valid):, 0])
        backtesting_actual = latest_y_train.iloc[len(y_lagged_valid)-len(X_lagged_valid)+lag_X:, 0].to_list()
        rmsfe.append(round(mean_squared_error(backtesting_predictions, backtesting_actual) ** 0.5, 3))
    print(rmsfe)

    # region 
    # Now that you have the optimal lags for each variable, create the final lagged DataFrame
    # final_lagged_X = pd.DataFrame(index=real_time_X.index)
    # for var, lag in optimal_lags.items():
    #     final_lagged_X[f'{var}_lag_{lag}'] = real_time_X[var].shift(lag)

    # Fill rows with -999 resulting from lagging
    # final_lagged_X.fillna(-999, inplace=True)
    # final_real_time_y = real_time_y.reindex(final_lagged_X.index)

    # Fit the final ADL model with the optimal lags for each variable
    # final_model = sm.OLS(final_real_time_y, sm.add_constant(final_lagged_X)).fit()
    # final_model = sm.OLS(final_real_time_y, final_lagged_X).fit()

    # Display the model summary
    # print(final_model.summary())

    # Display the optimal lags
    # print("Optimal lags for each variable:")
    # for var, lag in optimal_lags.items():
    #    print(f"{var}: {lag}")
    # endregion

    # region
    # forecast_results = {}

    # # Loop over each lag to build separate models and perform forecasts
    # for lag in range(1, 9):
    #     # Prepare the dataset for the current lag
    #     # Shift Y_t to align with data from t-lag for predictors
    #     X_lagged = real_time_X.shift(lag)
    #     y_lagged = real_time_y.shift(lag)
        
    #     # Drop rows with NaN resulting from shifting
    #     valid_indices = y_lagged.dropna().index
    #     X_lagged_valid = X_lagged.loc[valid_indices]
    #     y_lagged_valid = y_lagged.dropna()
    #     print(X_lagged_valid, y_lagged_valid)
        
    #     # Add constant to the predictors
    #     # X_lagged_const = sm.add_constant(X_lagged_valid, has_constant='add')
        
    #     # Fit the model
    #     model_lagged = sm.OLS(y_lagged_valid, X_lagged_valid).fit()
        
    #     # Prepare the most recent data point for forecasting Yt+lag|t
    #     print(real_time_X)
    #     X_recent = real_time_X.iloc[-lag:]  # Get the row corresponding to t-lag
    #     print(X_recent)
    #     # X_recent_const = sm.add_constant(X_recent, has_constant='add')
        
    #     # Perform the forecast
    #     forecasted_Yt_plus_lag = model_lagged.predict(X_recent)
    #     print(forecasted_Yt_plus_lag)
        
    #     # Store the forecast result
    #     forecast_results[f'Yt+{lag}|t'] = forecasted_Yt_plus_lag.iloc[0]

    # # Display forecast results
    # y_pred = []
    # for forecast_horizon, value in forecast_results.items():
    #     y_pred.append(value)
    #     #print(f"{forecast_horizon}: {value}")
    # endregion
        
    CI = [0.57, 0.842, 1.282] #50, 60, 80% predictional interval
    y_pred = pd.Series(y_pred)
    y_pred.index = latest_y_test.index
    print(y_pred)
    real_time_y.index = latest_y_train.index
    plot = plot_forecast_real_time(real_time_y[1:], y_pred, latest_y_test, CI, "ADL Model")
    
    return plot, y_pred, rmsfe
        
# ADL_MODEL("2012", "2")