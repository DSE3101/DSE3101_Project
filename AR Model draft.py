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
starting_year = "1990"
starting_quarter = "1"
ending_year = "2005"
ending_quarter = "4"
start_time = "ROUTPUT" + starting_year[-2:] + "Q" + starting_quarter
end_time = "ROUTPUT" + ending_year[-2:] + "Q" + ending_quarter
real_time_data = data[end_time].dropna()
real_time_data = real_time_data[((int(starting_year)-1947)*4):] #filters dataset to what the user specified

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
#print(optimal_lags)

#forming real time model
real_time_optimal_model = AutoReg(real_time_data, lags=optimal_lags)
real_time_model_fit = real_time_optimal_model.fit()
#print(real_time_model_fit.summary())

# forecast
forecasted_values = real_time_model_fit.predict(start=len(real_time_data), end=len(real_time_data)+10) # forecast 10 period ahead
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

###############################################
## using vintage data ##

latest_year = "2024"
latest_quarter = "1"
#start_time = "ROUTPUT" + starting_year[-2:] + "Q" + starting_quarter
#end_time = "ROUTPUT" + ending_year[-2:] + "Q" + ending_quarter
latest_time = "ROUTPUT" + latest_year[-2:] + "Q" + latest_quarter
latest_vintage_data = data[latest_time].dropna()
revised_vintage_data = latest_vintage_data[((int(starting_year)-1947)*4):((int(ending_year)-1947)*4)]

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

# forecasting
vintage_forecasted_values = vintage_model_fit.predict(start=len(real_time_data), end=len(real_time_data)+10) # forecast 10 period ahead
print(vintage_forecasted_values)
#print(type(vintage_forecasted_values))
print(revised_vintage_data) 

# function to plot forecasted values
def plot_forecast(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Revised Vintage Data', color='green')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.title('AR Model Forecast with Vintage data')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()
plot_forecast(revised_vintage_data, vintage_forecasted_values)


#transform the model by using entity-demeaned OLS regression (fixed effects) for adl 
#AR dont need bc no other entities/variables, just lags



