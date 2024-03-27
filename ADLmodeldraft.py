import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic
from statsmodels.graphics.tsaplots import plot_acf

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

#merge based on date
data_frames = [gdp, csr_conf, cpi, h_starts, ir_3m, ir_10y, m1, pcpi, consumption, exports, govt_spending, imports, investments,unemployment_rate]
merged_data = pd.concat(data_frames, axis=1)

#clean merged data
merged_data = merged_data.replace(',', '', regex=True)
merged_data = merged_data.replace(' ', '', regex=True)
merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
merged_data = merged_data.dropna()

#user to choose variables 
#ok wait i confused they choose variables and date range is it 

#adjust df so that it contains only the variables and lags
selected_data = merged_data

#choose minimum AIC based on lag lengths
min_aic = float('inf')
optimal_lag = 0

for lag in range(1, 11):  #can adjust range 
    model = VAR(merged_data)
    results = model.fit(lag)
    current_aic = aic(results.aic)
    
    if current_aic < min_aic:
        min_aic = current_aic
        optimal_lag = lag

#check autocorrelation
plot_acf(merged_data) 

#create ADL model with the optimal lag length
final_model = VAR(merged_data)
final_results = final_model.fit(optimal_lag)
print(final_results.summary())







