import pandas as pd
import numpy as np
from GetData import get_data
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from PlotGraphs import plot_forecast_real_time


def AR_MODEL(year_input, quarter_input):
    h_step_forecast = []
    h_step_lag = []
    rmsfe = []
    for h_step in range(8):
        real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
        real_time_y.reset_index(inplace=True)
        real_time_y.drop("index", axis=1, inplace=True)
        for j in range(1, 9):
            lag_col = real_time_y.shift(j+h_step).iloc[:, 0]
            real_time_y[f'yt-{j+h_step}'] = lag_col
        real_time_y.fillna(-999, inplace=True)
        real_time_y = real_time_y.iloc[9+h_step:, :]
        real_time_y.rename(columns={real_time_y.columns[0]: "yt"}, inplace=True)

        # Define the maximum number of lags
        k_max = 8
        # Initialize a list to store mean squared errors for each lag value
        mse_list = []
        for k in range(1, k_max + 1):
            # Prepare data for training
            X = real_time_y.iloc[:, 1:k+1].values  # Select lagged columns as features
            y = real_time_y['yt'].values  # Select the original column as target
            # Perform LOOCV
            loocv_mse = 0
            for i in range(len(real_time_y)):
                X_train = X[np.arange(len(real_time_y)) != i]
                y_train = y[np.arange(len(real_time_y)) != i]
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
        #print("Optimal number of lags:", optimal_lag)

        # Forecast Yt+i
        linear_model = LinearRegression()
        linear_model.fit(real_time_y.iloc[:, 1:optimal_lag+1], real_time_y.iloc[:, 0])
        X_predict = real_time_y.iloc[-1:, 1:optimal_lag+1]
        h_step_forecast.append(linear_model.predict(X_predict)[0])

        # Backtesting
        backtest_predictions = []
        for i in range(len(real_time_y)):
            backtest_X = real_time_y.iloc[i:i+1, 1:optimal_lag+1]
            backtest_predictions.append(linear_model.predict(backtest_X)[0])
        backtest_actual = latest_y_train.iloc[9+h_step:, 0].to_list()
        rmsfe.append(mean_squared_error(backtest_predictions, backtest_actual) ** 0.5)
    print(rmsfe)

    # Plots
    CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    h_step_forecast = pd.Series(h_step_forecast) #adding this into output
    h_step_forecast.index = latest_y_test.index
    
    temp = real_time_X.iloc[len(latest_y_test)*2:, 0]
    y_plot = real_time_y.iloc[:, 0]
    y_plot.index = temp.index

    real_time_plot = plot_forecast_real_time(y_plot, h_step_forecast, latest_y_test, CI, "AR Model")

    return h_step_forecast, h_step_lag, rmsfe, real_time_plot

#AR_MODEL("1981", "2")