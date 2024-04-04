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
from RandomForest import *
import pandas as pd
import pickle

with open('preprocessed_data.pkl', 'rb') as f:
    (RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT) = pickle.load(f)

macro_variables = [RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT]

# Create YYQq to slice vintages by index
vintages = ["65Q4"]
vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
vintages.extend([f'{i}Q{j}' for i in range(10, 24) for j in range(1, 5)])
vintages.extend(["24Q1"])

# year_input = input("Choose real time data from 1966 to 2023")
year_input = "2012"
# quarter_input = input("Choose a quarter from 1 to 4")
quarter_input = "2"

# Combine all variables for chosen quarter and remove rows
def get_vintage_data(year, quarter):
    quarter_index = vintages.index(f'{year[-2:]}Q{quarter}')
    df = []
    for var in macro_variables:
        df.append(var.iloc[:, quarter_index])
    df = pd.concat(df, axis=1)
    # Remove rows after chosen quarter
    df = df[df.index <= f"{year}:Q{quarter}"]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def MLTab():
    
    MLTab = html.Div([
        html.H3("Machine Learning Analysis"),
        html.P("In this section, we will use your selected training time period, coupled with our selected variables, to run a regression forest."),
        html.H4("Feature Importance"),
    ])

    return MLTab