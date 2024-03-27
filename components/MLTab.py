import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# test with real time first
# test with 60 most recent quarters
# err some 2023Q4 some 2024Q1
# just testing random forest first, remember to fix the dates
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


def MLTab():
    MLTab = [
        html.Strong("Test!"),
        html.P("Content for Real Time vs Vintage Data.")
        ]
    return MLTab