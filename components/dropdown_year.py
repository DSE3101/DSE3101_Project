import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')

def dropdown_year():
    date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')
    dropdown = dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year.year), 'value': str(year.year)} for year in date_range_yearly],
        value=str(date_range_yearly[-10].year),
        clearable=False,
        searchable=True,
        style={'width': '100%', 'color': 'black'},
    )
    return dropdown
