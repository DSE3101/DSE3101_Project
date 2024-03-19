import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from data import get_date_range

date_range = get_date_range()

date_range_yearly = pd.date_range(start='1960-01-01', end='2023-12-31', freq='YS')
timestamps = date_range.view('int64') // 10**9

def slider():
    slider = dcc.Slider(id='date-slider',
                            min=0,
                            max=len(date_range_yearly)-1,
                            value=len(date_range_yearly)-10,
                            marks = {i: {'label': str(year.year)[-2:]} for i, year in enumerate(date_range_yearly)},
                            step=1  # Assuming you want to step through each year
                            )
    return slider