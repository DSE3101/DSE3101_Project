import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

date_range_yearly = pd.date_range(start='1965-01-01', end='2023-12-31', freq='YS')

def dropdown_year():
    date_range_yearly = pd.date_range(start='1965-01-01', end='2019-10-01', freq='YS')
    dropdown = dcc.Dropdown(
        id = "dropdown-year",
        options=[{'label': str(year.year), 'value': str(year.year)} for year in date_range_yearly],
        value = "1970",
        clearable=False,
        searchable=True,
        style={'width': '50%', 'color': 'black'},
        placeholder="Select Year for Training",
    )
    return dropdown
