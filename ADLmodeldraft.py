#TRYING THE MODEL but i need help on the data part hurhur


import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from itertools import combinations

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

# i use gdp, cpi, imports, and consumption first
df = [gdp_latest,cpi_latest,imports_latest,consumption_latest]
selected_data = pd.concat(df, axis=1)
selected_data = selected_data.replace(',', '', regex=True)
selected_data = selected_data.replace(' ', '', regex=True)
selected_data = selected_data.astype(float)
print(selected_data)

#initialise best model and best aic
best_model = None
best_aic = np.inf
best_combi = None

#iterate thru all possible combinations of models to find best model based on lowest aic
for i in range(1, len(selected_data) + 1):
    for combi in itertools.combinations(selected_data, i):
        #combine the selected variables w gdp
        model_data = pd.concat([gdp_latest] + list(combi), axis = 1)

        #fit ADL model
        model = VAR(model_data)
        fitted_model = model.fit()

        #calc aic
        aic = model_fitted.aic

        #update best model if aic is lower
        if aic < best_aic:
            best_aic = aic
            best_model = model_fitted
            best_combi = combi

print("best combination:", best_combi)
print("best AIC:", best_aic)
print(fitted_model)



#check autocorrelation for each variable separately
for column in selected_data.columns:
    plot_acf(selected_data[column], title=f'Autocorrelation for {column}')
    plt.show()

# forecast 8 steps
forecasted_values = best_model.forecast(selected_data.values[-best_model.k_ar:], 8)

CI = [0.57, 0.842, 1.282] # 50, 60, 80% predictional interval

def plot_forecast(data, forecast, CI):
    plt.figure(figsize=(20,7))
    plt.plot(data.index, data['gdp'].values, label='Unrevised Real Time Data', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', color='red')
    for i, ci in enumerate(CI):
        alpha = 0.5 * (i + 1) / len(CI)
        lower_bound = forecast - ci * forecast.std()
        upper_bound = forecast + ci * forecast.std()
        plt.fill_between(forecast.index, lower_bound, upper_bound, color='blue', alpha=alpha)
    plt.title('ADL Model Forecast Test')
    plt.xlabel('Year:Quarter')
    plt.ylabel('rGDP')
    plt.legend()
    plt.show()

plot_forecast(selected_data, forecasted_values[:, 0], CI)





