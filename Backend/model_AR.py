import pandas as pd
import numpy as np
from Backend.GetData import get_data
from sklearn.linear_model import LinearRegression
from Backend.PlotGraphs import *

def AR_MODEL(year_input, quarter_input):
    h_steps = 8
    rmse = []
    forecasts = []
    optimal_lags = []
    for step in range(h_steps):
        # Read in data
        real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
        real_time_y.reset_index(drop=True, inplace=True)
        real_time_y.rename(columns={real_time_y.columns[0]: "yt"}, inplace=True)
        # Add 4+step NaN rows to prevent truncation of lagged columns
        real_time_y = pd.concat([real_time_y, pd.DataFrame([np.NaN] * (4+step), columns=['yt'])], ignore_index=True)
        # Add lagged columns; for Yt+i|t, add (Yt-i, Yt-i-1, Yt-i-2, Yt-i-3)
        for lag in range(1, h_steps+1):
            lag_col = real_time_y.iloc[:, 0]
            lag_col.index += (lag+step)
            real_time_y[f'yt-{lag+step}'] = lag_col
        # print(real_time_y) here to see what it looks like
        # Store the latest full row of lags for the final forecast Yt+i|t
        X_predict = real_time_y.iloc[[len(real_time_X)+step], 1:]
        # Drop NaN to ensure same sample set for all tests
        real_time_y.dropna(inplace=True)

        # Determine optimal no. of lags for each Yt+i|t model using MSE LOOCV
        mse_values = []
        for lag in range(1, h_steps+1):
            X = real_time_y.iloc[:, 1:lag+1].values
            y = real_time_y.iloc[:, 0].values
            residuals_squared = []
            for i in range(len(y)):
                # Leave one row out as test set, rest as trainig set
                X_train = X[np.arange(len(y)) != i]
                X_test = X[i].reshape(1, -1)
                y_train = y[np.arange(len(y)) != i]
                y_test = y[i]
                # Fit the LOO model
                model = LinearRegression()
                model.fit(X_train, y_train)
                # Predict
                prediction = model.predict(X_test)
                # Get residuals
                residuals_squared.append((y_test - prediction) ** 2)
            # Get MSE of residuals
            mse_values.append((sum(residuals_squared) / len(y))[0])

        # Optimal no. of lags = min MSE
        lowest_mse = min(mse_values)
        optimal_lag = mse_values.index(lowest_mse) + 1
        optimal_lags.append(optimal_lag)

        # RMSE for optimal lag model for Yt+i|t
        rmse.append(lowest_mse ** 0.5)
        #print(rmse)
        # Make Yt+i|t forecast
        linear_model = LinearRegression()
        # Fit using optimal lags
        linear_model.fit(real_time_y.iloc[:, 1:optimal_lag+1], real_time_y.iloc[:, 0])
        # Forecast using optimal lags
        X_predict = X_predict.iloc[:, :optimal_lag]
        # Store forecasted value
        forecasts.append(linear_model.predict(X_predict)[0])

    # Plot
    forecasts = pd.Series(forecasts)
    forecasts.index = latest_y_test.index
    PI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    _, real_time_y, _, _, _, _, _, _ = get_data(year_input, quarter_input)
    plot = plot_forecast_real_time(real_time_y.iloc[1:], forecasts, latest_y_test, PI, "AR Model", rmse)

    return forecasts, optimal_lags, rmse, plot

#AR_MODEL("1970", "1")