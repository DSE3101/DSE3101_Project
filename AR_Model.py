import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from GetData import get_data
from sklearn.metrics import mean_squared_error
import plotly.tools as tls

#need to run GetData.py first

def AR_MODEL(year_input, quarter_input):
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)

    def converting_to_stationary(y_data):
        y_data[y_data.columns[0]] = y_data[y_data.columns[0]].replace(-999,np.nan)
        data = y_data.dropna()
        return data

    def finding_minimum_aic(y_data):
        max_lags = 8 # since our data is quarterly, can consider up to 6-8 max lags
        aic_values = []
        for lag in range(1, max_lags + 1):
            if max_lags > lag:
                model = AutoReg(y_data[:-(max_lags-lag)], lags=lag)
                results = model.fit()
                aic_values.append(results.aic)
            else:
                model = AutoReg(y_data, lags=lag)
                results = model.fit()
                aic_values.append(results.aic)
        optimal_lag = np.argmin(aic_values) + 1 
        return optimal_lag

    def forming_AR_model(y_data, optimal_lags):
        optimal_model = AutoReg(y_data, lags=optimal_lags) 
        model_fit = optimal_model.fit()
        return model_fit

    def autocorrelation_plot(y_data):
        plot_acf(y_data, lags=8) # depends how many lags do yall want to display, chosen via max lags used
        #plt.show()

    def adfuller_stats(y_data):
        result = adfuller(y_data)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def forecasted_values_data(y_data, ar_model_fit):
        forecasted_values = ar_model_fit.predict(start=len(y_data), end=len(y_data)+11) #forecasting 12 periods ahead
        return forecasted_values

    def h_step_forecast(forecast_data):
        return forecast_data.values

    CI = [0.57, 0.842, 1.282] #50, 60, 80% predictional interval

    def plot_forecast_real_time(data, forecast, CI):
        fig, ax = plt.subplots(figsize=(3,3))  # Create a new figure and set the size
        ax.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
        for i, ci in enumerate(CI):
            alpha = 0.5 * (i + 1) / len(CI)
            lower_bound = forecast - ci * forecast.std()
            upper_bound = forecast + ci * forecast.std()
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=30, offset=0))
        ax.set_title('AR Model Forecast with Real-Time Data')
        ax.set_xlabel('Year:Quarter')
        ax.set_ylabel('rGDP')
        ax.legend()
        fig.savefig('assets/ar_real_time_plot.png')  # Save the figure explicitly
        plt.close(fig)  # Close the figure to prevent it from displaying


    def plot_forecast_vintage(data, forecast, CI):
        fig, ax = plt.subplots(figsize=(3,3))  # Create a new figure and set the size
        ax.plot(data.index, data.values, label='Revised Vintage Data', color='blue')
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
        for i, ci in enumerate(CI):
            alpha = 0.5 * (i + 1) / len(CI)
            lower_bound = forecast - ci * forecast.std()
            upper_bound = forecast + ci * forecast.std()
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=30, offset=0))
        ax.set_title('AR Model Forecast with Revised Vintage Data')
        ax.set_xlabel('Year:Quarter')
        ax.set_ylabel('rGDP')
        ax.legend()
        fig.savefig('assets/ar_vintage_plot.png')  # Save the figure explicitly
        plt.close(fig)  # Close the figure to prevent it from displaying



    def calculating_rmsfe(y_true, y_predicted):
        rmsfe = mean_squared_error(y_true,y_predicted)**(0.5)
        return rmsfe

    ###### for real time data ######
    real_time_data = converting_to_stationary(real_time_y)
    real_time_optimal_lags = finding_minimum_aic(real_time_data)
    real_time_AR_model = forming_AR_model(real_time_data,real_time_optimal_lags)
    autocorrelation_plot(real_time_data)
    adfuller_stats(real_time_data)
    realtime_table_of_forecasts = forecasted_values_data(real_time_data, real_time_AR_model)
    h_realtime = h_step_forecast(forecasted_values_data(real_time_data, real_time_AR_model)) 
    plot_forecast_real_time(real_time_data, realtime_table_of_forecasts, CI)
    real_time_rmsfe = calculating_rmsfe(latest_y_test,h_realtime)
    
    ###### for vintage data ######
    vintage_data = converting_to_stationary(latest_y_train)
    vintage_optimal_lags = finding_minimum_aic(vintage_data)
    vintage_AR_model = forming_AR_model(vintage_data,vintage_optimal_lags)
    autocorrelation_plot(vintage_data)
    adfuller_stats(vintage_data)
    vintage_table_of_forecasts = forecasted_values_data(vintage_data, vintage_AR_model)
    h_vintage = h_step_forecast(forecasted_values_data(vintage_data, vintage_AR_model)) 
    plot_forecast_vintage(vintage_data, vintage_table_of_forecasts, CI)
    vintage_rmsfe = calculating_rmsfe(latest_y_test,h_vintage)
    
    print('Lags chosen for real time AR model:',real_time_optimal_lags)
    print('Forecasted values for real time AR model:',h_realtime)
    print('Real time RMSFE:',real_time_rmsfe)
    print('Lags chosen for vintage AR model:',vintage_optimal_lags)
    print('Forecasted values for vintage AR model:',h_vintage)
    print('Vintage RMSFE:',vintage_rmsfe)

    return real_time_optimal_lags, h_realtime, real_time_rmsfe, vintage_optimal_lags, h_vintage, vintage_rmsfe

# Example usage
#AR_MODEL("2012","2")