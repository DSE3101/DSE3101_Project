import pandas as pd
import numpy as np
from Backend.GetData import get_data
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from Backend.PlotGraphs import *
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# def plot_forecast_real_time(data, forecast, actual, CI, modelName, rmse_values):
#         actual = actual.iloc[:,0]
#         fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

#         # Plotting the unrevised real-time data
#         ax.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
#         # Plotting the forecast
#         ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
#         # Plotting the actual data
#         ax.plot(actual.index, actual.values, color='green', label='Actual Data', alpha=0.6)
        
#         for i, ci in enumerate(CI):
#             alpha = 0.5 * (i + 1) / len(CI)
#             lower_bound = forecast - ci * rmse_values[i]
#             upper_bound = forecast + ci * rmse_values[i]
#             ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)

#         ax.set_xlim([data.index[0], forecast.index[-1]])
#         ax.xaxis.set_major_locator(MaxNLocator(5))
#         ax.set_title(f'{modelName} Forecast with Real-Time Data')
#         ax.set_xlabel('Year:Quarter')
#         ax.set_ylabel('Change in growth rate')
#         ax.legend()

#         plt.show()

def RANDOM_FOREST(year_input, quarter_input):
    # region Test for optimal n_estimators
    # real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
    # # Define a range of n_estimators values to try
    # n_estimators_range = [5, 10, 15, 20, 30]
    # # Initialize empty lists to store results
    # mean_scores = []
    # std_scores = []
    # # Perform grid search with cross-validation
    # for n_estimators in n_estimators_range:
    #     rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    #     scores = cross_val_score(rf, real_time_X, real_time_y.values.ravel(), cv=5, scoring='neg_mean_squared_error')
    #     mean_scores.append(np.mean(scores))
    #     std_scores.append(np.std(scores))
    # print(mean_scores, std_scores)
    # # Plot the mean scores
    # plt.errorbar(n_estimators_range, mean_scores, yerr=std_scores, fmt='-o', color='b')
    # plt.xlabel('n_estimators')
    # plt.ylabel('Mean Squared Error')
    # plt.title('Grid Search Results')
    # plt.grid(True)
    # plt.show()
    # endregion

    n_estimators = 10
    h_steps = 8
    top_n_variables = 6
    forecasts = []
    rmse = []

    for step in range(1, 1+h_steps):
        # Read in data
        real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)
        # Save last row Xt for forecasting Yt+i|t
        X_predict = real_time_X.iloc[-1:, :]
        # Lag X columns
        real_time_X_lagged = real_time_X.shift(step)
        real_time_X_lagged.dropna(inplace=True)
        # Match y rows
        real_time_y = real_time_y.iloc[-len(real_time_X_lagged):]

        # Train RF Model with all variables
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_model.fit(real_time_X_lagged, real_time_y.values.ravel())
        # Choose 6 most important variables
        feature_importance = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': real_time_X_lagged.columns, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        selected_variables = feature_importance_df.head(top_n_variables)['Feature'].tolist()
        # Train new RF Model with selected variables
        rf_model_selected = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_model_selected.fit(real_time_X_lagged[selected_variables], real_time_y.values.ravel())
        # Forecast Yt+i|t
        forecasts.append(rf_model_selected.predict(X_predict[selected_variables])[0])

        # Backtesting for Random Forest
        residuals_squared = []
        for i in range(len(real_time_y)):
            # Prepare LOO train test set
            loocv_X_train = real_time_X_lagged[selected_variables].reset_index(drop=True, inplace=False)
            loocv_X_train = loocv_X_train[np.arange(len(real_time_y)) != i]
            loocv_X_test = real_time_X_lagged[selected_variables].reset_index(drop=True, inplace=False)
            loocv_X_test = loocv_X_test.iloc[i, :]
            loocv_y_train = real_time_y.reset_index(drop=True, inplace=False)
            loocv_y_train = loocv_y_train[np.arange(len(real_time_y)) != i]
            loocv_y_test = real_time_y.reset_index(drop=True, inplace=False)
            loocv_y_test = loocv_y_test.iloc[i, :]
            # Fit the OLS model
            loocv_model = sm.OLS(loocv_y_train, loocv_X_train).fit()
            # Make the prediction
            loocv_prediction = loocv_model.predict(loocv_X_test.values)
            # Calculate residual
            residuals_squared.append((loocv_prediction - loocv_y_test) ** 2)
        # Append to rmse
        rmse.append(((sum(residuals_squared) / len(real_time_y))[0]) ** 0.5)

    # Plot
    forecasts = pd.Series(forecasts)
    forecasts.index = latest_y_test.index
    PI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    _, real_time_y, _, _, _, _, _, _ = get_data(year_input, quarter_input)
    plot = plot_forecast_real_time(real_time_y, forecasts, latest_y_test, PI, "RF Model", rmse)

    return rmse, plot, forecasts

# RANDOM_FOREST("2012", "2")