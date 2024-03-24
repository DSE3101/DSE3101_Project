import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import aic
from tkinter import *

# Load data
data = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")

# Convert data to numeric format
data = data.apply(pd.to_numeric, errors='coerce')

# Function to perform entity demeaning
def entity_demean(data):
    entities = data.index.str.extract(r'(\d+):Q(\d+)').astype(int)
    mean_by_entity = data.groupby(entities[0]).transform('mean')
    demeaned_data = data - mean_by_entity
    return demeaned_data

# Function to select optimal number of lags using AIC
def select_lags_AIC(data, max_lags=10):
    nobs = len(data)
    best_aic = np.inf
    best_lag = 0
    
    for lag in range(1, min(max_lags, nobs)):
        model = AutoReg(data, lags=lag)
        results = model.fit()
        curr_aic = aic(results)
        
        if curr_aic < best_aic:
            best_aic = curr_aic
            best_lag = lag
    
    return best_lag

# Function to fit AR model with optimal lags
def fit_AR_model(data, num_lags):
    model = AutoReg(data, lags=num_lags)
    results = model.fit()
    return results

# Function to convert quarterly data into datetime format
def quarter_to_datetime(quarter_str):
    year, quarter = map(int, quarter_str.split(':'))
    month = 3 * quarter  # Assuming quarters are represented as 1, 2, 3, 4
    return pd.Timestamp(year, month, 1)

# Function to handle button click event
def forecast():
    start_year_quarter = start_entry.get()
    end_year_quarter = end_entry.get()
    
    # Convert user input to datetime objects
    start_date = quarter_to_datetime(start_year_quarter)
    end_date = quarter_to_datetime(end_year_quarter)
    
    # Subset the data based on user input
    subset_data = data.loc[start_date:end_date]
    
    # Perform entity demeaning
    demeaned_data = entity_demean(subset_data)
    
    # Select optimal number of lags using AIC
    num_lags = select_lags_AIC(demeaned_data)
    
    # Fit AR model with optimal lags
    ar_model = fit_AR_model(demeaned_data, num_lags)
        
    # Forecast
    forecasted_values = ar_model.predict(start=len(demeaned_data), end=len(demeaned_data)+30)  # Forecast 30 periods ahead
        
    # Plot forecasted values
    plot_forecast(subset_data, forecasted_values)

# Function to plot forecasted values
def plot_forecast(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Actual Data', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.title('AR Model Forecast')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()

# Create GUI
root = Tk()
root.title("rGDP Forecasting Tool")

# Labels and entry widgets for start and end quarters
Label(root, text="Start Year:Quarter (YYYY:Qn):").grid(row=0, column=0, padx=10, pady=5)
start_entry = Entry(root)
start_entry.grid(row=0, column=1, padx=10, pady=5)
start_entry.insert(0, 'YYYY:Qn')

Label(root, text="End Year:Quarter (YYYY:Qn):").grid(row=1, column=0, padx=10, pady=5)
end_entry = Entry(root)
end_entry.grid(row=1, column=1, padx=10, pady=5)
end_entry.insert(0, 'YYYY:Qn')

# Button to trigger forecast
Button(root, text="Forecast", command=forecast).grid(row=2, columnspan=2, padx=10, pady=10)

root.mainloop()
