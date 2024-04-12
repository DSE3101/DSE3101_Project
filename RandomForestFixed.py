import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from GetData import get_data
from PlotGraphs import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def random_forest(year,quarter):
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year, quarter)
    top_n_variables = len(real_time_X.columns) // 10

    # region
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

    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

    y_pred = []
    rmsfe = []
    
    for i in range(1,9):
        # Loop 1, use xt-1 for a values to find yt, then append to pred
        # loop 2, use xt-2 for all values to find yt, then append to pred

        # Copy the DataFrame to avoid changing the original unexpectedly
        real_time_X_lagged = real_time_X.copy()
        # Calculate the start index for NaN replacement
        start_index = len(real_time_X) - i
        # Replace the last i rows' values with NaN
        real_time_X_lagged.iloc[start_index:] = np.nan
        real_time_X_lagged = real_time_X_lagged.shift(i)
        real_time_X_lagged = real_time_X_lagged.iloc[i:]
        real_time_Y_chopped = real_time_y[i:]

        # lag x by i, then fit into model
        rf_model.fit(real_time_X_lagged, real_time_Y_chopped.values.ravel())
        
        # Select top 6
        feature_importance = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': real_time_X_lagged.columns, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        selected_variables = feature_importance_df.head(top_n_variables)['Feature'].tolist()
        
        # Random forest with the top 6 again
        rf_model_selected = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model_selected.fit(real_time_X_lagged[selected_variables], real_time_Y_chopped.values.ravel())
        
        # Forecast yt_j
        y_pred.append(rf_model_selected.predict(real_time_X_lagged[selected_variables].iloc[-1:, :])[0])

        # Backtesting
        backtesting_predictions = []
        for row in range(len(real_time_X_lagged)):
            backtesting_predictions.append(rf_model_selected.predict(real_time_X_lagged[selected_variables].iloc[row:row+1, :])[0])
        backtesting_actual = latest_y_train.iloc[i:, 0].to_list()
        rmsfe.append(round(mean_squared_error(backtesting_predictions, backtesting_actual) ** 0.5,3))
    print(rmsfe)
    
    # Plots
    CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    y_pred = pd.Series(y_pred) #adding this into output
    y_pred.index = latest_y_test.index
    real_time_plot = plot_forecast_real_time(real_time_y[1:], y_pred, latest_y_test, CI, "RF Model")

    return rmsfe, real_time_plot, y_pred

# random_forest("2012", "3")