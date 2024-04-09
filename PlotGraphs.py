import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def plot_forecast_real_time(data, forecast, CI, modelName):
        fig, ax = plt.subplots(figsize=(4,4))  # Create a new figure and set the size
        ax.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
        for i, ci in enumerate(CI):
            alpha = 0.5 * (i + 1) / len(CI)
            lower_bound = forecast - ci * forecast.std()
            upper_bound = forecast + ci * forecast.std()
            ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)

        ax.set_xlim([data.index[-20], forecast.index[-1]])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(f'{modelName} Forecast with Real-Time Data')
        ax.set_xlabel('Year:Quarter')
        ax.set_ylabel('rGDP')
        ax.legend()

        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        
        image_png = buffer.getvalue()
        base64_string = base64.b64encode(image_png).decode('utf-8')
        buffer.close()
        
        return f"data:image/png;base64,{base64_string}"


def plot_forecast_vintage(data, forecast, CI, modelName):
    fig, ax = plt.subplots(figsize=(4,4))  # Create a new figure and set the size
    ax.plot(data.index, data.values, label='Revised Vintage Data', color='blue')
    ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
    for i, ci in enumerate(CI):
        alpha = 0.5 * (i + 1) / len(CI)
        lower_bound = forecast - ci * forecast.std()
        upper_bound = forecast + ci * forecast.std()
        ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)
    
    ax.set_xlim([data.index[-20], forecast.index[-1]])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(f'{modelName} Forecast with Revised Vintage Data')
    ax.set_xlabel('Year:Quarter')
    ax.set_ylabel('rGDP')
    ax.legend()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    
    image_png = buffer.getvalue()
    base64_string = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    
    return f"data:image/png;base64,{base64_string}"