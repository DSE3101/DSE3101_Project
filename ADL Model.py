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
# Columns of testing set {var}24Q1, is renamed to follow columns of training set {var}12Q2
# This is to allow the ARDL model to match column names
latest_X_test.columns = real_time_X.columns
latest_y_test.columns = real_time_y.columns

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
            #x_data_subset = x_data_subset.iloc[1:,:]
            
            # Fit ARDL model with different lag orders
            for lag in range(1, max_lag + 1):
                ardl_model = ARDL(endog=real_time_y, lags=lag, exog=x_data_subset, order=lag)
                ardl_results = ardl_model.fit()
                aic = ardl_results.aic
                
                # Update best model if current model has lower AIC
                if aic < best_aic:
                    best_aic = aic
                    best_model = (ardl_model, ardl_results, vars_combination, lag)
                    best_x_cols = x_data_subset.columns
    print("Best Model:")
    print("Variables:", best_model[2])
    print("Lag Order:", best_model[3])
    print("AIC:", best_model[1].aic)
    print(best_model[1].summary())
    return best_model, best_x_cols

def combining_data(x_data,y_data,chosen_variables):
    formatted_chosen_variables =[]
    for i in chosen_variables:
        formatted_chosen_variables.append(f'{i}{year_input[-2:]}Q{quarter_input}')
    x = x_data.loc[:,formatted_chosen_variables]
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

candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']
best_model, best_x_cols = best_adl_model(candidate_vars)
variables_in_realtime_model = best_model[2]
real_time_optimal_lags = best_model[3]
forecast_steps = 12
forecast = best_model[1].forecast(steps=12, exog=latest_X_test.loc[:, best_x_cols])
print(forecast)


forecast.index = latest_y_test.index
rmsfe = mean_squared_error(forecast, latest_y_test[f"ROUTPUT{year_input[-2:]}Q{quarter_input}"]) ** 0.5
print('rmsfe:',rmsfe)
######## acf & adf #########
combined = combining_data(real_time_X,real_time_y,variables_in_realtime_model)
combined_data = converting_to_stationary(combined)
#plot_individual_acf(combined_data)
#plot_adl_autocorrelation(combined_data)
#adf_test(combined_data)



y = real_time_y.iloc[1:, 0]
y.index = y.index.strftime('%YYYY:%Q')
# Plot the forecast against time
plt.figure(figsize=(10, 6))
plt.plot(y.index, y.values, label='Unrevised Real Time Data', color='blue')
plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(y.index[::30]) 
plt.title('ADL Model Forecast with Real-Time Data')
plt.legend()
plt.show()