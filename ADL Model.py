import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
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

def best_adl_model(variables):
    max_lag = 8
    best_model = None
    best_aic = float('inf')
    for num_vars in range(1, len(variables) + 1):
        for vars_combination in combinations(variables, num_vars):
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
    print("Best Model:")
    print("Variables:", best_model[2])
    print("Lag Order:", best_model[3])
    print("AIC:", best_model[1].aic)
    print(best_model[1].summary())
    return best_model

def combining_data(x_data,y_data):
    x = x_data.loc[:,[f'M1{year_input[-2:]}Q{quarter_input}',f'HSTARTS{year_input[-2:]}Q{quarter_input}',f'IPM{year_input[-2:]}Q{quarter_input}',f'OPH{year_input[-2:]}Q{quarter_input}']]
    combined = pd.concat([y_data,x],axis=1)
    return combined

def converting_to_stationary(y_data):
    y_data[y_data.columns[0]] = y_data[y_data.columns[0]].replace(-999,np.nan)
    data = y_data.dropna()
    return data

def plot_individual_acf(data):
    for variable in data.columns:
        autocorrelation_plot(data.iloc[:,data.columns.get_loc(variable)],label=variable)
        plt.legend()
        plt.show()

def plot_adl_autocorrelation(data):
    for variable in data.columns:
        autocorrelation_plot(data[variable],label=variable)
    plt.legend()
    plt.show()

def adf_test(data):
    for variable in data.columns:
        adf_result = adfuller(data[variable])
        print('For variable:',variable)
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print(f'   {key}: {value}')

real_time_X.index = real_time_X.index.map(convert_to_datetime)
real_time_y.index = real_time_y.index.map(convert_to_datetime)

candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH',]
best_model = best_adl_model(candidate_vars)
variables_in_realtime_model = best_model[2]
real_time_optimal_lags = best_model[3]

combined = combining_data(real_time_X,real_time_y)
combined_data = converting_to_stationary(combined)
#plot_individual_acf(combined_data)
#plot_adl_autocorrelation(combined_data)
#adf_test(combined_data)
#print(best_model[1].forecast(steps=12))

y = combined_data.iloc[:, 0]  # Assuming the endogenous variable is in the first column
x = combined_data.iloc[:, 1:]
forecast_steps = 12
forecast = best_model[1].predict(start=len(y), end=len(y) + forecast_steps - 1)

# Plot the forecasted values along with the actual values
plt.figure(figsize=(10, 6))
plt.plot(y.index, y.values, label='Actual', marker='o')
plt.plot(pd.date_range(start=y.index[-1], periods=forecast_steps + 1, freq='Q')[1:], forecast, label='Forecast', marker='o')
plt.title('ARDL Forecast')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

'''
# Define the start and end points for the forecast period
start = len(real_time_y)
end = start + 11  # Forecasting 12 steps ahead

# Make the forecast
forecast = best_model[1].predict(start=start, end=end, )

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(real_time_y.index, real_time_y.values, label='Actual Data', color='blue')
plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
plt.title('ARDL Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
'''




'''
def plot_autocorrelation_resid(adl_results):
    plot_acf(adl_results.resid, lags=8) 
    plt.title('Autocorrelation Plot of Residuals')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
#plot_autocorrelation_resid(best_model[1])

def adf_test_(adl_results):
    adf_result = adfuller(adl_results)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value}')
#adf_test(best_model[1])

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

'''
