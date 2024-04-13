import pandas as pd
import numpy as np
from GetData import get_data
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from PlotGraphs import plot_forecast_real_time
# import matplotlib.pyplot as plt


def AR_MODEL(year_input, quarter_input):
    h_step_forecast = []
    h_step_lag = []
    rmse = []
    # Loop for 8 steps
    for h_step in range(8):
        # Read in data
        real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
        # AR model only needs Yt
        real_time_y.reset_index(inplace=True)
        real_time_y.drop("index", axis=1, inplace=True)
        real_time_y.rename(columns={real_time_y.columns[0]: "yt"}, inplace=True)
        # Add NaN rows so we don't lose rows when shifting
        # Add 8+h_steps as each model tests for up to 8 lags
        real_time_y = pd.concat([real_time_y, pd.DataFrame([np.NaN] * (8+h_step), columns=['yt'])], ignore_index=True)

        # To forecast Yt+i|t, we need Yt, Yt-i, Yt-i-1, ..., Yt-i-8
        for j in range(1, 9):
            # Adds the respective lagged columns
            lag_col = real_time_y.iloc[:, 0]
            lag_col.index += (j+h_step)
            real_time_y[f'yt-{j+h_step}'] = lag_col
        real_time_y.fillna(-999, inplace=True)
        # Remove the top 8+h_step rows of NaN due to lagging
        real_time_y = real_time_y.iloc[8+h_step:, :]

        # Determine optimal number of lags to forecast Yt+i|t using MSE LOOCV
        k_max = 8 # Max number of lags
        mse_list = []
        for k in range(1, k_max + 1):
            # Prepare data for training
            X = real_time_y.iloc[:len(real_time_y)-(8+h_step), 1:k+1].values  # Select lagged columns as features
            y = real_time_y['yt'].values[:len(real_time_y)-(8+h_step)]  # Select the original column as target
            # Perform LOOCV
            loocv_mse = 0
            for i in range(len(y)):
                X_train = X[np.arange(len(y)) != i]
                y_train = y[np.arange(len(y)) != i]
                X_test = X[i].reshape(1, -1)
                y_test = y[i]
                # Fit model
                model = LinearRegression()
                model.fit(X_train, y_train)
                # Predict and calculate MSE
                y_pred = model.predict(X_test)
                loocv_mse += mean_squared_error([y_test], [y_pred])
            # Calculate average MSE for this lag value
            avg_mse = loocv_mse / len(real_time_y)
            mse_list.append(avg_mse)
        # Find the lag value with the minimum average MSE
        optimal_lag = mse_list.index(min(mse_list)) + 1
        h_step_lag.append(optimal_lag)

        # Forecast Yt+i
        linear_model = LinearRegression()
        # We shifted all lags without truncating, resulting in the bottom 8+h_step rows having NaN (in a triangle)
        # Remove those bottom NaN rows
        linear_model.fit(real_time_y.iloc[:len(real_time_y)-(8+h_step), 1:optimal_lag+1], real_time_y.iloc[:len(real_time_y)-(8+h_step), 0])
        # Predict on the t+1 row (this row contains the columns Yt-i, Yt-i-1, ...) to obtain Yt+i|t
        X_predict = real_time_y.iloc[-(8+h_step):-(8+h_step)+1, 1:optimal_lag+1]
        # Store the forecasted value
        h_step_forecast.append(linear_model.predict(X_predict)[0])

        # Backtesting
        # Remove the bottom NaN triangle and use the last 4 values as pseudo-out-of-sample
        # Training set is up till the last 4 values
        backtest_X_train = real_time_y.iloc[:(-4-8-h_step), 1:optimal_lag+1]
        backtest_y_train = real_time_y.iloc[:(-4-8-h_step), 0]
        backtest_model = LinearRegression()
        backtest_model.fit(backtest_X_train, backtest_y_train)
        # Perform pseudo out-of-sample forecasting using the final 4 observations
        pseudo_out_of_sample_X = real_time_y.iloc[(-4-8-h_step):(-8-h_step), 1:optimal_lag+1]
        pseudo_out_of_sample_forecast = backtest_model.predict(pseudo_out_of_sample_X)
        print(f"Forecast for step {h_step+1}: {pseudo_out_of_sample_forecast}")
        # Calculate rmse for the pseudo out-of-sample forecasting
        backtest_actual = real_time_y.iloc[(-4-8-h_step):(-8-h_step), 0].to_list()
        rmse.append(round(mean_squared_error(pseudo_out_of_sample_forecast, backtest_actual) ** 0.5, 7))
    # Prepare output for plotting
    h_step_forecast = pd.Series(h_step_forecast)
    h_step_forecast.index = latest_y_test.index
    rmse = pd.Series(rmse)
    rmse.index = h_step_forecast.index

    _ , plot_real_time_y, _, _, _, plot_latest_y_test, _, _ = get_data(year_input, quarter_input)
    print(plot_real_time_y, plot_latest_y_test)
    print(h_step_forecast)
    print(rmse)

    # Plots
    CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval

    y_plot = plot_real_time_y.iloc[:(8+h_step), 0]
    real_time_plot = plot_forecast_real_time(y_plot, h_step_forecast, latest_y_test, CI, "AR Model")

    return h_step_forecast, h_step_lag, rmse, real_time_plot

AR_MODEL("2002", "2")