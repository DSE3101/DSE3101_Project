import pandas as pd
import numpy as np
from GetData import get_data
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Helper function to calculate LOOCV MSE for a single variable at a given lag
def loocv_mse_for_variable(X, y):
    mse_values = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)
    return np.mean(mse_values)

def plot_forecast_real_time(data, forecast, actual, CI, modelName, rmse_values):
        actual = actual.iloc[:,0]
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

        # Plotting the unrevised real-time data
        ax.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
        # Plotting the forecast
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
        # Plotting the actual data
        ax.plot(actual.index, actual.values, color='green', label='Actual Data', alpha=0.6)
        
        for i, ci in enumerate(CI):
            alpha = 0.5 * (i + 1) / len(CI)
            lower_bound = forecast - ci * rmse_values[i]
            upper_bound = forecast + ci * rmse_values[i]
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)

        ax.set_xlim([data.index[0], forecast.index[-1]])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(f'{modelName} Forecast with Real-Time Data')
        ax.set_xlabel('Year:Quarter')
        ax.set_ylabel('Change in growth rate')
        ax.legend()

        plt.show()

def ADL_MODEL(year_input, quarter_input, ar_optimal_lags):
    h_steps = 8
    forecasts = []
    rmse = []
    # Get data
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
    # Define candidate variables
    candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']
    # Match to vintage naming convention
    candidate_vars = [f'{var}{year_input[-2:]}Q{quarter_input}' for var in candidate_vars]
    # Filter candidate variables from overall data
    real_time_X = real_time_X.loc[:, candidate_vars]

    # Find optimal no. of lags for each variable
    '''
    We assume all variables will be chosen, only need to find optimal no. of lags for each variable
    because choosing all combinations of variables is too computationally expensive.
    
    Another problem with this model is it determines the no. of lags of each variable sequentially.
    I.e., The optimal no. of lags of CPI is chosen assuming no lags of the other variables
    Next, the optimal no. of lags of RUC is chosen, it may then affect the optimal no. of lags of CPI.
    However, we will not check for this as it is too computationally expensive.
    '''
    optimal_lags = {}
    # Iterate over each variable to find its optimal number of lags
    for var in real_time_X.columns:
        best_mse = np.inf
        best_lag = 0
        for lag in range(1, h_steps + 1):
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

    # Prepare lagged data
    for lag_X in range(1, h_steps+1):
        # Prepare lagged X data
        X_lagged_valid = pd.DataFrame(index=real_time_X.index)
        # Prepare X_predict for final forecast Yt+i|t
        X_predict = []
        # Add optimal lag of each X variable to X_lagged_valid
        for var, optimal_lag in optimal_lags.items():
            # Store Xt with optimal lag
            X_predict.append(real_time_X[var].iloc[-optimal_lag])
            # Lagged X variables, let it truncate
            lagged_series = real_time_X[var].shift(optimal_lag)
            X_lagged_valid[f'{var}_lag_{optimal_lag}'] = lagged_series

        # Lag all X variables by the additional lag_X
        X_lagged_valid = X_lagged_valid.shift(lag_X-1)
        # Add Yt with optimal lag and additional lag_X
        X_lagged_valid["y_lagged"] = real_time_y.shift(lag_X+ar_optimal_lags[lag_X-1]-1)
        # Add Yt after lagging to X_predict
        X_predict.append(real_time_y.iloc[-ar_optimal_lags[lag_X-1], 0])
        # Drop rows with NaN resulting from lagging
        X_lagged_valid.dropna(inplace=True)
        # Prepare Yt column
        y_train = real_time_y.iloc[len(real_time_y)-len(X_lagged_valid):]

        # Backtesting
        residuals_squared = []
        for i in range(len(y_train)):
            # Prepare LOO train test set
            loocv_X_train = X_lagged_valid.reset_index(drop=True, inplace=False)
            loocv_X_train = loocv_X_train[np.arange(len(y_train)) != i]
            loocv_X_test = X_lagged_valid.reset_index(drop=True, inplace=False)
            loocv_X_test = loocv_X_test.iloc[i, :]
            loocv_y_train = y_train.reset_index(drop=True, inplace=False)
            loocv_y_train = loocv_y_train[np.arange(len(y_train)) != i]
            loocv_y_test = y_train.reset_index(drop=True, inplace=False)
            loocv_y_test = loocv_y_test.iloc[i, :]
            # Fit the OLS model
            loocv_model = sm.OLS(loocv_y_train, loocv_X_train).fit()
            # Make the prediction
            loocv_prediction = loocv_model.predict(loocv_X_test.values)
            # Calculate residual
            residuals_squared.append((loocv_prediction - loocv_y_test) ** 2)
        # Append to rmse
        rmse.append(((sum(residuals_squared) / len(y_train))[0]) ** 0.5)

        # Fit the model using OLS
        model_lagged = sm.OLS(y_train, X_lagged_valid).fit()
        # Perform the forecast
        forecasts.append(model_lagged.predict(X_predict)[0])

    # Plot
    forecasts = pd.Series(forecasts)
    forecasts.index = latest_y_test.index
    CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    _, real_time_y, _, _, _, _, _, _ = get_data(year_input, quarter_input)
    plot = plot_forecast_real_time(real_time_y, forecasts, latest_y_test, CI, "ADL Model", rmse)
    
    return plot, forecasts, rmse

# ADL_MODEL("2012", "2", [1, 1, 1, 1, 1, 1, 1, 1])