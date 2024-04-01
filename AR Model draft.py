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
from fanchart import fan

data = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")

#example of user input
ending_year = "2005"
ending_quarter = "4"
end_time = "ROUTPUT" + ending_year[-2:] + "Q" + ending_quarter
real_time_data = data[end_time].diff().dropna() #filters dataset to what the user specified and takes first difference

# must decide whats the smallest no. of lags you want to give the model without reducing sample size 
max_lags = 8 # i asked denis and he said since our data is quarterly, can consider up to 6-8 max lags

# fit AutoReg models with different lag values
aic_values = []
for lag in range(1, max_lags + 1):
    if max_lags > lag:
        model = AutoReg(real_time_data[:-(max_lags-lag)], lags=lag)
        results = model.fit()
        aic_values.append(results.aic)
    else:
        model = AutoReg(real_time_data, lags=lag)
        results = model.fit()
        aic_values.append(results.aic)

# choose the lag order that minimizes the AIC
optimal_lags = np.argmin(aic_values) + 1 
#print(aic_values)
print('AIC value for real-time data:', aic_values[optimal_lags-1])
print('Optimal number of lags for real-time data:', optimal_lags)

#forming real time model
real_time_optimal_model = AutoReg(real_time_data, lags=20)
real_time_model_fit = real_time_optimal_model.fit()
#print(real_time_model_fit.summary())

# autocorrelation_plot(real_time_data) & ADF statistic
plot_acf(real_time_data, lags=max_lags) ## erm how many lags do yall want to display
plt.show()
real_time_result = adfuller(real_time_data)
print('ADF Statistic: %f' % real_time_result[0])
print('p-value: %f' % real_time_result[1])
print('Critical Values:')
for key, value in real_time_result[4].items():
 print('\t%s: %.3f' % (key, value))

# forecast 8 periods ahead (can change)
forecasted_values = real_time_model_fit.predict(start=len(real_time_data), end=len(real_time_data)+20) 
print(forecasted_values)


#probs = [0.05, 0.20, 0.35, 0.65,0.80,  0.95]
#fan(data=real_time_data,probs=probs,history= forecasted_values.index >= 1)
''' just ignore the red stuff first
# attempt 1
probs = [0.05, 0.20, 0.35, 0.65, 0.80, 0.95]
def plot_fan_chart_forecast(data, forecasted_values, probs, confidence_level=0.80):
    # Define the confidence level for the fan chart
    confidence_level = confidence_level

    # Compute the standard deviation of the forecast errors
    forecast_errors = forecasted_values - data.iloc[-1]  # Use iloc to access the last element
    forecast_error_std = np.std(forecast_errors)

    # Plot the original data
    plt.plot(data.index, data.values, label='Original Data', color='blue')

    # Iterate over each probability level and plot the corresponding shaded area
    for prob in probs:
        # Compute upper and lower bounds for the given probability
        upper_bound = forecasted_values + forecast_error_std * np.sqrt(10) * prob
        lower_bound = forecasted_values - forecast_error_std * np.sqrt(10) * prob

        # Compute the alpha value for gradient shading based on the probability level
        alpha = min(1, max(0, prob*2 - 0.1))  # Adjust the scaling factor as needed

        # Plot the shaded area with gradient shading
        plt.fill_between(forecasted_values.index, lower_bound, upper_bound, color='red', alpha=alpha,
                         label='Fan Chart ({}% CI)'.format(int(confidence_level*100)))

    # Add labels and legend
    plt.title('Fan Chart Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Example usage:
plot_fan_chart_forecast(real_time_data, forecasted_values, probs)
'''

'''
# attempt 2
# Plot the fan chart
def plot_forecast_with_fan_chart(data, forecast, upper_bound, lower_bound):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.fill_between(forecast.index, lower_bound, upper_bound, color='red', alpha=0.3, label='Fan Chart ({}% CI)'.format(int(confidence_level*100)))
    plt.title('AR Model Forecast with Real-Time Data (Fan Chart)')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()

# Plot the forecast with the fan chart
plot_forecast_with_fan_chart(real_time_data, forecasted_values, upper_bound, lower_bound)
'''

#ATTEMPT 3
y = forecasted_values.values
x = forecasted_values.index
CI = [0.842, 1.036, 1.282, 1.96] #60, 70, 80, 95% confidence interval
# function to plot forecasted values
def plot_forecast(data, forecast, CI):
    plt.figure(figsize=(20,7))
    plt.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    for i, ci in enumerate(CI):
        alpha = 0.5 * (i + 1) / len(CI)
        lower_bound = forecast - ci * forecast.std()
        upper_bound = forecast + ci * forecast.std()
        plt.fill_between(forecast.index, lower_bound, upper_bound, color='green', alpha=alpha)
    plt.title('AR Model Forecast with Real-Time Data')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()

plot_forecast(real_time_data, forecasted_values, CI)


######################### using vintage data #########################

latest_year = "2024"
latest_quarter = "1"
latest_time = "ROUTPUT" + latest_year[-2:] + "Q" + latest_quarter
latest_vintage_data = data[latest_time].diff().dropna()
revised_vintage_data = latest_vintage_data[:((int(ending_year)-1947)*4)]
max_vintage_lags = 8 

# fit AutoReg models with different lag values
vintage_aic_values = []
for lag in range(1, max_lags + 1):
    if max_lags > lag:
        model = AutoReg(revised_vintage_data[:-(max_lags-lag)], lags=lag)
        results = model.fit()
        vintage_aic_values.append(results.aic)
    else:
        model = AutoReg(revised_vintage_data, lags=lag)
        results = model.fit()
        vintage_aic_values.append(results.aic)
# choose the lag order that minimizes the AIC
optimal_vintage_lags = np.argmin(vintage_aic_values) + 1 
#print(vintage_aic_values)
print('AIC value for vintage data:', vintage_aic_values[optimal_vintage_lags-1])
print('Optimal number of lags for vintage data:', optimal_vintage_lags)

#forming AR model
vintage_optimal_model = AutoReg(revised_vintage_data, lags=optimal_vintage_lags)
vintage_model_fit = vintage_optimal_model.fit()

# autocorrelation_plot(revised_vintage_data) & ADF statistic
plot_acf(revised_vintage_data, lags=max_lags) ## erm how many lags do yall want to display
plt.show()
vintage_result = adfuller(revised_vintage_data)
print('ADF Statistic: %f' % vintage_result[0])
print('p-value: %f' % vintage_result[1])
print('Critical Values:')
for key, value in vintage_result[4].items():
 print('\t%s: %.3f' % (key, value))

# forecasting 8 periods ahead
vintage_forecasted_values = vintage_model_fit.predict(start=len(revised_vintage_data), end=len(revised_vintage_data)+8)

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



