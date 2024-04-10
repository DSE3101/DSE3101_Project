import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from GetData import get_data
from PlotGraphs import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def random_forest(year, quarter):
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year, quarter)
    top_n_variables = len(real_time_X.columns) // 3

    '''
    No need train test split because we are comparing against actual values from latest vintage directly
    '''

    # Train Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(real_time_X, real_time_y.values.ravel())

    # Get feature importance scores
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': real_time_X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Choose top N most important variables
    selected_variables = feature_importance_df.head(top_n_variables)['Feature'].tolist()
    print(f"Top {top_n_variables} Variables Selected:")
    print(selected_variables)

    # Expanding window
    y_pred = []
    mse_loocv_values = []
    # h step forecast
    for i in range(8):

        # MSE LOOCV for each model, but takes very long to train
        # def mse_loocv(model, X, y):
        #     n = X.shape[0]
        #     mse_values = []
        #     for i in range(n):
        #         # Remove the i-th sample
        #         X_train = X.drop(X.index[i])
        #         y_train = y.drop(y.index[i])
        #         # Fit the model on the training data
        #         model.fit(X_train, y_train.values.ravel())
        #         # Predict the removed sample
        #         y_pred = model.predict(X.iloc[[i]])
        #         # Calculate the squared error for this sample
        #         mse_values.append((y_pred - y.iloc[i])**2)
        #         print((y_pred - y.iloc[i])**2)
        #     # Calculate the mean of squared errors
        #     mse_loocv = np.mean(mse_values)
        #     return mse_loocv
        
        real_time_X_loop, real_time_y_loop, latest_X_train_loop, latest_y_train_loop, latest_X_test_loop, latest_y_test_loop, curr_year, curr_quarter = get_data(year, quarter)
        # Adjust selected variables names to match each next quarter and year
        temp = []
        for var in selected_variables:
            temp.append(f'{var[:-4]}{year[-2:]}Q{quarter}')
        selected_variables = temp
        # Train model and make 1 step forecast
        rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_selected.fit(real_time_X_loop[selected_variables], real_time_y_loop.values.ravel())
        y_pred.append(rf_model_selected.predict(real_time_X_loop[selected_variables].iloc[-1:, :])[0])
        # Step to next vintage
        quarter = str(int(quarter) + 1)
        if quarter > "4":
            year = str(int(year) + 1)
            quarter = "1"
        
        # MSE LOOCV for each iterated model but takes very long to run
        # mse_loocv_values.append(mse_loocv(rf_model_selected, real_time_X_loop[selected_variables], real_time_y_loop))
        # print(mse_loocv)
        

    # RMSFE of 1 step ahead forecast
    rmsfe = mean_squared_error(y_pred, latest_y_test) ** 0.5
    print(f'RMSFE: {rmsfe}')
    '''
    We are making one-step ahead forecasts iteratively, where each prediction depends only on the previous data and not on the entire dataset.
    LOOCV typically involves training the model on all but one data point and then validating on the omitted data point.
    However, in our iterative forecasting setup, each prediction is made sequentially and the training data continually changes.
    Therefore, we use the metric RMSFE to assess the performance of the model.
    '''

    # Plots
    CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval
    
    y_pred = pd.Series(y_pred) #adding this into output
    y_pred.index = latest_y_test.index
    real_time_plot = plot_forecast_real_time(real_time_y[1:], y_pred, CI, "RF Model")
    latest_plot = plot_forecast_vintage(latest_y_train[1:], latest_y_test.iloc[:, 0], CI, "RF Model")

    selected_variables_final = [var[:-4] for var in selected_variables]
    importance_values = feature_importance_df["Importance"].tolist()
    selected_variables_importance_dict = dict(zip(selected_variables_final, importance_values))
    return selected_variables_importance_dict, rmsfe, real_time_plot, latest_plot, y_pred

random_forest("2012", "2")