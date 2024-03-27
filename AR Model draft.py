import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import aic
from tkinter import *
from tkinter import ttk
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

data = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")

#example of user input
ending_year = "2005"
ending_quarter = "4"
end_time = "ROUTPUT" + ending_year[-2:] + "Q" + ending_quarter
real_time_data = data[end_time].diff().dropna() #filters dataset to what the user specified and takes first difference


# define the maximum number of lags based on the number of observations
max_lags = min(30, len(real_time_data) - 1)  #limiting to a maximum of 30 lags or (nobs - 1)
# must decide whats the smallest no. of lags you want to give the model without reducing sample size 
# fit AutoReg models with different lag values
aic_values = []
for lag in range(10, max_lags + 1):
    model = AutoReg(real_time_data, lags=lag)
    results = model.fit()
    aic_values.append(results.aic)
# choose the lag order that minimizes the AIC
optimal_lags = np.argmin(aic_values) + 1 
#print(aic_values)
#print(optimal_lags)

#forming real time model
real_time_optimal_model = AutoReg(real_time_data, lags=optimal_lags)
real_time_model_fit = real_time_optimal_model.fit()
#print(real_time_model_fit.summary())

# autocorrelation_plot(real_time_data) & ADF statistic
plot_acf(real_time_data, lags=optimal_lags)
plt.show()
real_time_result = adfuller(real_time_data)
print('ADF Statistic: %f' % real_time_result[0])
print('p-value: %f' % real_time_result[1])
print('Critical Values:')
for key, value in real_time_result[4].items():
 print('\t%s: %.3f' % (key, value))

# forecast 10 periods ahead (can change)
forecasted_values = real_time_model_fit.predict(start=len(real_time_data), end=len(real_time_data)+10) 
print(real_time_data)
print(forecasted_values)
#print(type(forecasted_values))

# function to plot forecasted values
def plot_forecast(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.title('AR Model Forecast with Real-Time Data')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()
plot_forecast(real_time_data, forecasted_values)

######################### using vintage data #########################
latest_year = "2024"
latest_quarter = "1"
latest_time = "ROUTPUT" + latest_year[-2:] + "Q" + latest_quarter
latest_vintage_data = data[latest_time].diff().dropna()
revised_vintage_data = latest_vintage_data[:((int(ending_year)-1947)*4)]

max_vintage_lags = min(10, len(revised_vintage_data) - 1)  #limiting to a maximum of 10 lags or (nobs - 1)
# fit AutoReg models with different lag values
vintage_aic_values = []
for lag in range(1, max_vintage_lags + 1):
    model = AutoReg(revised_vintage_data, lags=lag)
    results = model.fit()
    vintage_aic_values.append(results.aic)
# choose the lag order that minimizes the AIC
optimal_vintage_lags = np.argmin(vintage_aic_values) + 1 

#forming AR model
vintage_optimal_model = AutoReg(revised_vintage_data, lags=optimal_vintage_lags)
vintage_model_fit = vintage_optimal_model.fit()

# autocorrelation_plot(revised_vintage_data) & ADF statistic
plot_acf(revised_vintage_data, lags=optimal_lags)
plt.show()
vintage_result = adfuller(revised_vintage_data)
print('ADF Statistic: %f' % vintage_result[0])
print('p-value: %f' % vintage_result[1])
print('Critical Values:')
for key, value in vintage_result[4].items():
 print('\t%s: %.3f' % (key, value))

# forecasting 10 periods ahead
vintage_forecasted_values = vintage_model_fit.predict(start=len(revised_vintage_data), end=len(revised_vintage_data)+10)

# function to plot forecasted values
def plot_vintage_forecast(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Revised Vintage Data', color='green')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.title('AR Model Forecast with Vintage Data')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()
plot_vintage_forecast(revised_vintage_data, vintage_forecasted_values)


#transform the model by using entity-demeaned OLS regression (fixed effects) for adl 
#AR dont need bc no other entities/variables, just lags



