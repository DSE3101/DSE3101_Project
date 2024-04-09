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

real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data("2017","1")
# Columns of testing set {var}24Q1, is renamed to follow columns of training set {var}12Q2
# This is to allow the ARDL model to match column names
year_input='2017'
quarter_input = '1'
print(latest_X_test)
print(latest_y_test)
print(real_time_X)
print(real_time_y)
print(latest_X_train)
print(latest_y_train)
latest_X_test.columns = latest_X_train.columns
latest_y_test.columns = latest_y_train.columns

def convert_to_datetime(year_quarter_str):
    year, quarter = year_quarter_str.split(':')    # Split year and quarter
    quarter_to_month = {'Q1': "1", 'Q2': "4", 'Q3': "7", 'Q4': "10"}    # Map quarter to month
    month = quarter_to_month[quarter]
    datetime_obj = pd.to_datetime(f'{year}-{month}-01')    # Create datetime object
    datetime_obj = pd.Period(datetime_obj, freq='Q')     # Specify frequency as quarterly
    return datetime_obj

def best_adl_model(variables,x_data,y_data):
    max_lag = 8
    best_model = None
    best_aic = float('inf')
    for num_vars in range(1, len(variables) + 1):
        for vars_combination in combinations(variables, num_vars):
            x_columns = [f'{var}{year_input[-2:]}Q{quarter_input}' for var in vars_combination] # Generate column names for the combination
            x_data_subset = x_data.loc[:, x_columns]            # Filter real_time_X for selected columns
        
            for lag in range(1, max_lag + 1):             # Fit ARDL model with different lag orders
                ardl_model = ARDL(endog=y_data, lags=lag, exog=x_data_subset, order=lag)
                ardl_results = ardl_model.fit()
                aic = ardl_results.aic
                
                if aic < best_aic:                # Update best model if current model has lower AIC
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

CI = [0.57, 0.842, 1.282] #50, 60, 80% predictional interval
def plot_forecast_real_time(data, forecast):
    y = data.iloc[1:, 0]
    y.index = y.index.strftime('%Y')
    plt.figure(figsize=(15, 6))
    plt.plot(y.index, y.values, label='Unrevised Real Time Data', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    for i, ci in enumerate(CI):
        alpha = 0.5 * (i + 1) / len(CI)
        lower_bound = forecast - ci * forecast.std()
        upper_bound = forecast + ci * forecast.std()
        plt.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(y.index[::40]) 
    plt.title('ADL Model Forecast with Real-Time Data')
    plt.legend()
    plt.show()

def calculating_rmsfe(y_predicted):
    y_predicted.index = latest_y_test.index
    rmsfe = mean_squared_error(y_predicted, latest_y_test[f"ROUTPUT{year_input[-2:]}Q{quarter_input}"]) ** 0.5
    print('rmsfe:',rmsfe)
    return rmsfe

####### real time #######
'''
real_time_X.index = real_time_X.index.map(convert_to_datetime)
real_time_y.index = real_time_y.index.map(convert_to_datetime)
candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']
best_realtime_model, best_x_cols_realtime = best_adl_model(candidate_vars,real_time_X,real_time_y)
variables_in_realtime_model = best_realtime_model[2]
real_time_optimal_lags = best_realtime_model[3]
forecast_steps = 12
real_time_forecast = best_realtime_model[1].forecast(steps=12, exog=latest_X_test.loc[:, best_x_cols_realtime])
print(real_time_forecast)
calculating_rmsfe(real_time_forecast)
plot_forecast_real_time(real_time_y,real_time_forecast)
    ## acf & adf ##
combined = combining_data(real_time_X,real_time_y,variables_in_realtime_model)
combined_data = converting_to_stationary(combined)
#plot_individual_acf(combined_data)
#plot_adl_autocorrelation(combined_data)
#adf_test(combined_data)
'''

####### vintage data #######
latest_X_train.index = latest_X_train.index.map(convert_to_datetime)
latest_y_train.index = latest_y_train.map(convert_to_datetime)
candidate_vars = ['CPI', 'RUC', 'M1', 'HSTARTS', 'IPM', 'OPH']
best_vintage_model, best_x_cols_vintage = best_adl_model(candidate_vars,latest_X_train,latest_y_train)
variables_in_vintage_model = best_vintage_model[2]
vintage_optimal_lags = best_vintage_model[3]
forecast_steps = 12
vintage_forecast = best_vintage_model[1].forecast(steps=12, exog=latest_X_test.loc[:, best_x_cols_vintage])
print(vintage_forecast)
calculating_rmsfe(vintage_forecast)
plot_forecast_real_time(latest_y_train,vintage_forecast)
    ## acf & adf ##
combined_vintage = combining_data(latest_X_train,latest_y_train,variables_in_vintage_model)
combined_data_vintage = converting_to_stationary(combined_vintage)
#plot_individual_acf(combined_data_vintage)
#plot_adl_autocorrelation(combined_data_vintage)
#adf_test(combined_data_vintage)




