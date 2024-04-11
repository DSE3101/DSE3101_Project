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


def AR_MODEL(year_input, quarter_input):
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year_input, quarter_input)

    def mse_loocv(model, X, y):
        n = X.shape[0]
        mse_values = []
        for i in range(n):
            # Remove the i-th sample
            X_train = X.drop(X.index[i])
            y_train = y.drop(y.index[i])
            # Fit the model on the training data
            model.fit(X_train, y_train.values.ravel())
            # Predict the removed sample
            y_pred = model.predict(X.iloc[[i]])
            # Calculate the squared error for this sample
            mse_values.append((y_pred - y.iloc[i])**2)
            # print((y_pred - y.iloc[i])**2)
        # Calculate the mean of squared errors
        mse_loocv = np.mean(mse_values)
        return mse_loocv
    
    for i in range(8):
        # Create lag cols
        for j in range(1, 9):
            test = real_time_y.shift(j)
            real_time_y[f"yt-{j}"] = pd.Series(test.iloc[:, 0])
        # Slice top 8     
        real_time_y = real_time_y.iloc[9:,:]
        real_time_y.fillna(-999, inplace=True)

        mse_loocv_values = []
        for j in range(1,9):
            # yt = real_time_y.iloc[:,0]
            loop_model = AutoReg(real_time_y.iloc[:, 0], lags = j)
            #mse_loocv_values.append(mse_loocv(loop_model, real_time_y.iloc[1:j, 0], real_time_y.iloc[:, 0]))
            
        #print(mse_loocv_values)


    # # Plots
    # CI = [0.57, 0.842, 1.282] #50, 60, 80% predictional interval
    # real_time_plot = plot_forecast_real_time(real_time_y[1:], y_pred, latest_y_test, CI, "AR Model")
    # real_time_rmsfe = calculating_rmsfe(y_pred, latest_y_test)
    # #latest_plot = plot_forecast_vintage(la, latest_y_test, CI, "AR Model")
    # print('Lags chosen for real time AR model:',real_time_optimal_lags)
    # print('Forecasted values for real time AR model:\n',y_pred)
    # print('Real time RMSFE:',real_time_rmsfe)

    #return real_time_optimal_lags, real_time_rmsfe, real_time_plot, y_pred

# Example usage
AR_MODEL("2012","2")