#ADL(1,1) 

from GetData import get_data
from itertools import combinations
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data()

print(real_time_X) 

# function to create lagged variables
def create_lag(X, y, lag):

    #lag the variables
    y_lagged = y.shift(lag).dropna()
    X_lagged = X.shift(lag).dropna()

    return X_lagged, y_lagged

# function to create best ADL model based on lowest AIC
def create_ADL(X, y):
    best_model = None
    best_aic = float('inf')
    best_combination = None
    
    for i in range(1, len(X.columns) + 1): 
        for combi in combinations(X.columns, i): #generate all possible combi of x variables of i length

            #combine y with the x variables
            model_data = pd.concat([y] + [X[list(combi)]], axis=1)
            
            # create ADL model
            model = VAR(model_data)
            model_fitted = model.fit(maxlags=1)
            
            #then calc the aic
            aic_score = model_fitted.aic
            
            # choose the model with the lowest AIC
            if aic_score < best_aic:
                best_aic = aic_score
                best_model = model_fitted
                best_combination = combi
                
    return best_model, best_combination

# create real time ADL model
real_time_X_lagged, real_time_y_lagged = create_lag(real_time_X, real_time_y, lag=1)
real_time_model, real_time_best_combination = create_ADL(real_time_X_lagged, real_time_y_lagged)

# create latest ADL model
latest_X_train_lagged, latest_y_train_lagged = create_lag(latest_X_train, latest_y_train, lag=1)
latest_model, latest_best_combination = create_ADL(latest_X_train_lagged, latest_y_train_lagged)

# forecast 8 steps ahead
real_time_forecast = real_time_model.forecast(real_time_X_lagged.values[-real_time_model.k_ar:], 8)
latest_forecast = latest_model.forecast(latest_X_train_lagged.values[-latest_model.k_ar:], 8)


# calc RSME of both 
real_time_rmse = rmse(real_time_forecast, latest_y_test)
latest_rmse = rmse(latest_forecast, latest_y_test)

# function to plot the forecasts
def plot_forecast_fanchart(data, forecast, CI, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='Actual', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    for ci in CI:
        lower_bound = forecast - ci * forecast.std()
        upper_bound = forecast + ci * forecast.std()
        plt.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=0.1)
    plt.title(title)
    plt.xlabel('Year:Quarter')
    plt.ylabel('GDP')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot the real time forecast
plot_forecast_fanchart(latest_y_test, real_time_forecast, [0.57, 0.842, 1.282], 'Real-Time Model Forecast')

# plot the latest forecast
plot_forecast_fanchart(latest_y_test, latest_forecast, [0.57, 0.842, 1.282], 'Latest Model Forecast')

# print RMSE
print("Real-Time Model RMSE:", real_time_rmse)
print("Latest Model RMSE:", latest_rmse)