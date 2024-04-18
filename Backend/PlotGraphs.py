import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from matplotlib.ticker import MaxNLocator

def plot_forecast_real_time(data, forecast, actual, PI, modelName, rmse_values):
    actual = actual.iloc[:,0]
    # data = pd.concat([data, (pd.DataFrame([forecast.iloc[0]], index=[forecast.index[0]], columns=[data.columns[0]]))])
    forecast = pd.concat([pd.Series(data.iloc[-1].values, index=[data.index[-1]]), forecast])
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

    # Plotting the unrevised real-time data
    ax.plot(data.index, data.values, label='Unrevised Real Time Data', color='blue')
    # Plotting the forecast
    ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
    # Plotting the actual data
    ax.plot(actual.index, actual.values, color='green', label='Actual Data', alpha=0.6)
    
    rmse_values = [0] + rmse_values
    rmse_values = pd.Series(rmse_values)
    rmse_values.index = forecast.index
    
    for i, pi in enumerate(PI):
        alpha = 0.5 * (i + 1) / len(PI)
        lower_bound = forecast - pi * rmse_values
        upper_bound = forecast + pi * rmse_values
        ax.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)

    ax.set_xlim([data.index[-12], forecast.index[-1]])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(f'{modelName} Forecast with Real-Time Data')
    ax.set_xlabel('Year:Quarter')
    ax.set_ylabel('Change in growth rate')
    ax.legend()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    # plt.show()
    buffer.seek(0)
    
    image_png = buffer.getvalue()
    base64_string = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    
    return f"data:image/png;base64,{base64_string}"
