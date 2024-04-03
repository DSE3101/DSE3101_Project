import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

## Obsolete, will remove when confirmed
#Checkbox component
def checkbox():
    checkbox = dcc.Checklist(
                id='checkbox-menu',
                options=[
                    {'label': 'Consumer Confidence', 'value': 'csr_conf'},
                    {'label': 'Consumer Price Index (CPI)', 'value': 'cpi'},
                    {'label': 'Housing Starts', 'value': 'hstarts'},
                    {'label': 'Interest Rates (3 Mths)', 'value': 'ir_3m'},
                    {'label': 'Interest Rates (10 years)', 'value': 'ir_10y'},
                    {'label': 'Money 1', 'value': 'm1'},
                    {'label': 'Producer Consumer Price Index', 'value': 'pcpi'},
                    {'label': 'Consumption', 'value': 'cons'},
                    {'label': 'Exports', 'value': 'exp'},
                    {'label': 'Government Spendings', 'value': 'govt_s'},
                    {'label': 'Imports', 'value': 'imp'},
                    {'label': 'Investments', 'value': 'inv'},
                    {'label': 'Unemployment Rates', 'value': 'unemp'},
                ],
                value=[]  # Default selected values
            )
    return checkbox