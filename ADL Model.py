import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from GetData import get_data
from sklearn.metrics import mean_squared_error
import pickle
import statsmodels.api as sm
from statsmodels.tsa.api import ARDL

real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data("2012","2")

x_columns = real_time_X.columns
# def data_transformation_x(x_data):
x_realtime = real_time_X.diff().dropna()
# def data_transformation_y(y_data):
y_realtime = real_time_y.diff().dropna()

print(x_realtime)
print(y_realtime)
ardl_model = ARDL(y_realtime, lags=8, exog=x_realtime,order=1)
ardl_model.fit()





'''

def finding_min_aic(y_data, x_data):
    max_lags = 8 # since our data is quarterly, can consider up to 6-8 max lags
    aic_values = []
    for lag in range(1, max_lags + 1):
        if max_lags > lag:
            x_lagged = sm.tsa.lagmat(x_data[:-(max_lags-lag)], maxlag=lag)
            y_lagged = sm.tsa.lagmat(y_data[:-(max_lags-lag)], maxlag=lag)
            adl_model = sm.OLS(y_lagged, x_lagged)
            adl_results = adl_model.fit()
            aic_values.append(adl_results.aic)
        else:
            x_lagged = sm.tsa.lagmat(x_data, maxlag=lag)
            y_lagged = sm.tsa.lagmat(y_data, maxlag=lag)
            adl_model = sm.OLS(y_lagged, x_lagged)
            adl_results = adl_model.fit()
            aic_values.append(adl_results.aic)
    optimal_lag = np.argmin(aic_values) + 1 
    #print(aic_values)
    return optimal_lag
'''
def forming_ADL_model(y_data,x_data,lag_y,lag_x):
    y_lagged = sm.tsa.lagmat(y_data, maxlag=lag_y, trim='both')
    x_lagged = sm.tsa.lagmat(x_data, maxlag=lag_x, trim='both')   
    y_lagged = y_lagged[:-(len(y_lagged)-lag_y)]
    x_lagged = x_lagged[:-(len(x_lagged)-lag_x)]   
    final_data = np.concatenate((y_lagged, x_lagged), axis=1)
    final_data = sm.add_constant(final_data) #intercept
    print(type(final_data))
    print(final_data)
    endogenous = final_data[(y_data).columns]
    exogenous = final_data.drop(columns=y_data.columns)
    adl_model = sm.OLS(endogenous, exogenous)
    adl_results = adl_model.fit()
    #print(adl_results.summary())
    return adl_results
    '''
def forming_ADL_model(y_data, x_data, lag_y, lag_x):
    # Generate lagged arrays for y and x variables
    y_lagged = sm.tsa.lagmat(y_data, maxlag=lag_y, trim='both')
    x_lagged = sm.tsa.lagmat(x_data, maxlag=lag_x, trim='both')

    # Convert lagged arrays to pandas DataFrames
    y_lagged_df = pd.DataFrame(y_lagged, index=y_data.index[lag_y:], columns=[f'y_lag_{i+1}' for i in range(lag_y)])
    x_lagged_df = pd.DataFrame(x_lagged, index=x_data.index[lag_x:], columns=[f'x_lag_{i+1}' for i in range(lag_x)])

    # Ensure that indices of both DataFrames match
    common_index = y_lagged_df.index.intersection(x_lagged_df.index)
    y_lagged_df = y_lagged_df.loc[common_index]
    x_lagged_df = x_lagged_df.loc[common_index]

    # Combine lagged y and x variables
    final_data = pd.concat([y_lagged_df, x_lagged_df], axis=1)

    # Add constant term
    final_data = sm.add_constant(final_data)

    # Split the data into endogenous (y) and exogenous (x) variables
    endogenous = final_data[y_data.name]
    exogenous = final_data.drop(columns=y_data.name)

    # Fit the OLS model
    adl_model = sm.OLS(endogenous, exogenous)
    adl_results = adl_model.fit()

    return adl_results


def plot_autocorrelation(adl_results):
    plot_acf(adl_results.residuals, lags=20)  # Adjust 'lags' as needed
    plt.title('Autocorrelation Plot of Residuals')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

def adf_test(adl_results):
    adf_result = adfuller(adl_results.residuals)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value}')

forecast_steps = 12
def forecasting(optimal_model):
    forecast = optimal_model.forecast(steps=forecast_steps)
    return forecast

def plot_real_time_forecast(forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(real_time_data.index, real_time_data.values, label='Actual')
    plt.plot(pd.date_range(start=real_time_data.index[-1], periods=forecast_steps+1, freq='Q')[1:], forecast, label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('rGDP')
    plt.title('Real Time ADL Model Forecast')
    plt.legend()
    plt.show()


optimal_lags = finding_min_aic(real_time_data,x_data)
print('Optimal lags:', optimal_lags)
adl_model = forming_ADL_model(real_time_data,x_data,optimal_lags,optimal_lags)
    





def find_min_aic(y_data, x_data):
    max_lags = 8  # Maximum number of lags to consider
    min_aic = float('inf')  # Initialize minimum AIC
    optimal_lags = None  # Initialize optimal lag configuration

    n_obs = len(y_data)  # Number of observations

    # Iterate over different lag orders for the endogenous variable (y)
    for lag_y in range(1, max_lags + 1):
        # Generate lagged y variable
        y_lagged = sm.tsa.lagmat(y_data, maxlag=lag_y)

        # Iterate over different lag orders for each exogenous variable (x)
        for lag_x in range(1, max_lags + 1):
            # Generate lagged versions of each exogenous variable
            x_lagged = sm.tsa.lagmat(x_data, maxlag=lag_x)

            # Determine the maximum lag order to ensure the same sample size
            max_lag = max(lag_y, lag_x)

            # Trim the lagged datasets to have the same sample size
            y_lagged_trimmed = y_lagged[max_lag:]
            x_lagged_trimmed = x_lagged[max_lag:]

            # Combine lagged y and x variables
            lagged_data = pd.concat([pd.DataFrame(y_lagged_trimmed), pd.DataFrame(x_lagged_trimmed)], axis=1)

            # Add constant term
            lagged_data = sm.add_constant(lagged_data)

            # Fit the OLS model
            model = sm.OLS(y_data[max_lag:], lagged_data)
            results = model.fit()

            # Check if current AIC is lower than minimum AIC
            if results.aic < min_aic:
                min_aic = results.aic
                optimal_lags = (lag_y, lag_x)
                optimal_model = results

    return optimal_lags, optimal_model

# Find optimal lag configuration and fit the ADL model
optimal_lags, optimal_model = find_min_aic(real_time_data, x_data)

# Print the optimal lag configuration
print("Optimal lag configuration:", optimal_lags)

# Print the summary of the optimal model
print("Optimal model summary:", optimal_model.summary())

'''
