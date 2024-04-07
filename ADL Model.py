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
from statsmodels.tsa.ardl import ARDLResults
from itertools import combinations

real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data("2012","2")
year_input='2012'
quarter_input = '2'
# x_data = real_time_X.loc[:,[f'CPI{year_input[-2:]}Q{quarter_input}',f'RUC{year_input[-2:]}Q{quarter_input}',f'M1{year_input[-2:]}Q{quarter_input}',f'HSTARTS{year_input[-2:]}Q{quarter_input}',f'IPM{year_input[-2:]}Q{quarter_input}',f'OPH{year_input[-2:]}Q{quarter_input}']]
# x_data = x_data.diff().dropna()

# x_columns = real_time_X.columns
# # def data_transformation_x(x_data):
# x_realtime = real_time_X.diff().dropna()
# # def data_transformation_y(y_data):
# y_realtime = real_time_y.diff().dropna()


def convert_to_datetime(year_quarter_str):
    # Split year and quarter
    year, quarter = year_quarter_str.split(':')
    # Map quarter to month
    quarter_to_month = {'Q1': "1", 'Q2': "4", 'Q3': "7", 'Q4': "10"}
    month = quarter_to_month[quarter]
    # Create datetime object
    datetime_obj = pd.to_datetime(f'{year}-{month}-01')
    # Specify frequency as quarterly
    datetime_obj = pd.Period(datetime_obj, freq='Q')
    return datetime_obj

real_time_X.index = real_time_X.index.map(convert_to_datetime)
real_time_y.index = real_time_y.index.map(convert_to_datetime)

# Define candidate variables
candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']

# Define maximum lag order
max_lag = 8

best_model = None
best_aic = float('inf')

# Iterate over all combinations of variables
for num_vars in range(1, len(candidate_vars) + 1):
    for vars_combination in combinations(candidate_vars, num_vars):
        # Generate column names for the combination
        x_columns = [f'{var}{year_input[-2:]}Q{quarter_input}' for var in vars_combination]
        
        # Filter real_time_X for selected columns
        x_data_subset = real_time_X.loc[:, x_columns]
        
        # Fit ARDL model with different lag orders
        for lag in range(1, max_lag + 1):
            ardl_model = ARDL(endog=real_time_y, lags=lag, exog=x_data_subset, order=lag)
            ardl_results = ardl_model.fit()
            aic = ardl_results.aic
            
            # Update best model if current model has lower AIC
            if aic < best_aic:
                best_aic = aic
                best_model = (ardl_model, ardl_results, vars_combination, lag)

# Print results of the best model
print("Best Model:")
print("Variables:", best_model[2])
print("Lag Order:", best_model[3])
print("AIC:", best_model[1].aic)
print(best_model[1].summary())



# ardl_model = ARDL(endog=y_realtime, lags=8, exog=x_data, order=8)
# # ardl_model = ARDL(y_realtime, lags=8, exog=x_realtime, order=2)
# ardl_results = ardl_model.fit()
# print(ardl_results.summary())
# print(ardl_model.ardl_order)
# print(ardl_results.aic)



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

'''
