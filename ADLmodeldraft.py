#from last week LOL its q shit
# m currently editing on it - fang 

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt

#DATA PREPROCESSING
# USED NICS DATA PREPROCESSING FOR THIS
# cigxm, gdp
def phil_month_to_quarter_cigxm(df):
    df["DATE"] = df["DATE"].str.replace(":", "-")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

# cpi, h_starts, m1, pcpi, unemployment_rate
def phil_month_to_quarter_others(df):
    df["DATE"] = df["DATE"].str.replace(":", "-")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.month % 3 == 0].copy()
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

def fred_month_to_quarter(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.month % 3 == 0].copy()
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

def fred_month_to_quarter_ir_3m(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

csr_conf = pd.read_csv("./data/project data/Consumer Confidence Indicator.csv")
csr_conf = fred_month_to_quarter(csr_conf)

cpi = pd.read_csv("./data/project data/CPI.csv")
cpi = phil_month_to_quarter_others(cpi)

h_starts = pd.read_csv("./data/project data/Housing Starts.csv")
h_starts = phil_month_to_quarter_others(h_starts)

ir_3m = pd.read_csv("./data/project data/Interest Rates 3M.csv")
ir_3m = fred_month_to_quarter_ir_3m(ir_3m)

ir_10y = pd.read_csv("./data/project data/Interest Rates 10Y.csv")
ir_10y = fred_month_to_quarter(ir_10y)

m1 = pd.read_csv("./data/project data/M1.csv")
m1 = phil_month_to_quarter_others(m1)

pcpi = pd.read_csv("./data/project data/PCPI.csv")
pcpi = phil_month_to_quarter_others(pcpi)

consumption = pd.read_csv("./data/project data/Real Consumption.csv")
consumption = phil_month_to_quarter_cigxm(consumption)

exports = pd.read_csv("./data/project data/Real Export.csv")
exports = phil_month_to_quarter_cigxm(exports)

govt_spending = pd.read_csv("./data/project data/Real Govt Spending.csv")
govt_spending = phil_month_to_quarter_cigxm(govt_spending)

imports = pd.read_csv("./data/project data/Real Import.csv")
imports = phil_month_to_quarter_cigxm(imports)

investments = pd.read_csv("./data/project data/Real Non-Residential Investment.csv")
investments = phil_month_to_quarter_cigxm(investments)

gdp = pd.read_csv("./data/project data/Real Output.csv")
gdp = phil_month_to_quarter_cigxm(gdp)

unemployment_rate = pd.read_csv("./data/project data/Unemployment Rate.csv")
unemployment_rate = phil_month_to_quarter_others(unemployment_rate)

#user choose date range
#USED NICS ONE ALSO. used last 60 rows, 24Q1
csr_conf_latest = csr_conf.iloc[-60:,-2].reset_index(drop=True)
cpi_latest = cpi.iloc[-60:,-2].reset_index(drop=True)
h_starts_latest = h_starts.iloc[-60:, -2].reset_index(drop=True)
ir_3m_latest = ir_3m.iloc[-60:, -2].reset_index(drop=True)
ir_10y_latest = ir_10y.iloc[-60:, -2].reset_index(drop=True)
m1_latest = m1.iloc[-60:, -2].reset_index(drop=True)
pcpi_latest = pcpi.iloc[-60:, -2].reset_index(drop=True)
consumption_latest = consumption.iloc[-60:, -2].reset_index(drop=True)
exports_latest = exports.iloc[-60:, -2].reset_index(drop=True)
govt_spending_latest = govt_spending.iloc[-60:, -2].reset_index(drop=True)
imports_latest = imports.iloc[-60:, -2].reset_index(drop=True)
investments_latest = investments.iloc[-60:, -2].reset_index(drop=True)
gdp_latest = gdp.iloc[-60:, -2].reset_index(drop=True)
unemployment_rate_latest = unemployment_rate.iloc[-60:, -2].reset_index(drop=True)

#user to choose variables 
# example they choose cpi, import, consumption
df = [gdp_latest,cpi_latest,imports_latest,consumption_latest]
selected_data = pd.concat(df, axis=1)
selected_data = selected_data.replace(',', '', regex=True)
selected_data = selected_data.replace(' ', '', regex=True)
selected_data = selected_data.astype(float)
print(selected_data)

aic_values = []
for lag in range(1, 20):  #choose optimum from 20 lags
    model = VAR(selected_data)
    results = model.fit(lag)
    aic_values.append(results.aic)
    
optimal_lags = np.argmin(aic_values) + 1 
#print(aic_values)
print(optimal_lags)

#check autocorrelation for each variable separately
for column in selected_data.columns:
    plot_acf(selected_data[column], title=f'Autocorrelation for {column}')
    plt.show()

#create ADL model with the optimal lag length
final_model = VAR(selected_data)
final_results = final_model.fit(optimal_lags)
#print(final_results.summary())
print(final_results)

# forecasting 10 periods ahead
# Get the index of the 'ROUTPUT24Q1' variable in the selected_data DataFrame
output_index = selected_data.columns.get_loc('ROUTPUT24Q1')
# Forecast only the 'ROUTPUT24Q1' variable for the next 10 periods
forecasted_values = final_results.forecast(y=selected_data.values[-optimal_lags:], steps=10)[:, output_index]

# Create a pandas DataFrame for the forecasted values with appropriate index
forecast_index = pd.period_range(selected_data.index[-1], periods=10, freq='Q')
forecasted_values_df = pd.DataFrame(forecasted_values, index=forecast_index, columns=['Forecast'])

# Plot the forecasted values
plot_vintage_forecast(selected_data['ROUTPUT24Q1'], forecasted_values_df)

# function to plot forecasted values
def plot_vintage_forecast(data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label='Original Data', color='green')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red')
    plt.title('ADL Model Forecast')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()






