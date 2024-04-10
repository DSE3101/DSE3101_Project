import pandas as pd
import numpy as np
import matplotlib
import warnings
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from GetData import get_data
from sklearn.metrics import mean_squared_error
import plotly.tools as tls
import base64
from io import BytesIO
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from PlotGraphs import *
from dm import *
matplotlib.use('Agg')


#need to run GetData.py first

def AR_MODEL(year_input, quarter_input):
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

    def calculating_rmsfe(y_true, y_predicted):
        rmsfe = mean_squared_error(y_true,y_predicted)**(0.5)
        return rmsfe

    # Get real time data
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)

    # Get optimal number of lags
    real_time_data = converting_to_stationary(real_time_y)
    real_time_data.reset_index(inplace=True)
    real_time_data = real_time_data.drop("index", axis=1)
    real_time_data = pd.Series(real_time_data.iloc[:, 0])
    real_time_optimal_lags = finding_minimum_aic(real_time_data)
    #autocorrelation_plot(real_time_data)
    #adfuller_stats(real_time_data)

    # Expanding window
    y_pred = []
    for i in range(8):
        # Train and predict the AR Model
        real_time_X_loop, real_time_y_loop, latest_X_train_loop, latest_y_train_loop, latest_X_test_loop, latest_y_test_loop, curr_year, curr_quarter = get_data(year_input, quarter_input)
        real_time_y_loop.reset_index(inplace=True)
        real_time_y_loop = real_time_y_loop.drop("index", axis=1)
        ar_model = forming_AR_model(real_time_y_loop, real_time_optimal_lags)
        y_pred.append(ar_model.predict(start=len(real_time_y_loop), end=len(real_time_y_loop)).iloc[0])
        # 1 step forward
        quarter_input = str(int(quarter_input) + 1)
        if quarter_input > "4":
            year_input = str(int(year_input) + 1)
            quarter_input = "1"
    y_pred = pd.Series(y_pred)
    y_pred.index = latest_y_test.index

    # Plots
    CI = [0.57, 0.842, 1.282] #50, 60, 80% predictional interval
    real_time_plot = plot_forecast_real_time(real_time_data, y_pred, latest_y_test, CI, "AR Model")
    real_time_rmsfe = calculating_rmsfe(y_pred, latest_y_test)
    #latest_plot = plot_forecast_vintage(la, latest_y_test, CI, "AR Model")
    #print('Lags chosen for real time AR model:',real_time_optimal_lags)
    #print('Forecasted values for real time AR model:\n',y_pred)
    #print('Real time RMSFE:',real_time_rmsfe)

    return real_time_optimal_lags, real_time_rmsfe, real_time_plot, y_pred

# Example usage
AR_MODEL("2012","2")