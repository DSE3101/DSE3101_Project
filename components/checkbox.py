import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

#Checkbox component
def checkbox():
    checkbox = dcc.Checklist(
                id='checkbox-menu',
                options=[
                    {'label': 'M1', 'value': 'opt2'},
                    {'label': 'Investment', 'value': 'opt3'},
                    {'label': 'Total Reserves', 'value': 'opt4'},
                    {'label': 'Nonborrowed Reserves', 'value': 'opt5'},
                    {'label': 'Nonborrowed Reserves + Extended Credit', 'value': 'opt6'},
                    {'label': 'Monetary Base', 'value': 'opt7'},
                    {'label': 'Civilian Unemployed Rate', 'value': 'opt8'},
                    {'label': 'CPI vs Chain-weighted Price Index', 'value': 'opt9'},
                    {'label': '3-month T Bill Rate', 'value': 'opt10'},
                    {'label': '10-year T-bond Rate', 'value': 'opt11'}
                ],
                value=[]  # Default selected values
            )
    return checkbox