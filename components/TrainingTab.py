import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from components.checkbox import checkbox
from components.dropdown_quarter import dropdown
from components.slider import slider
from data import mainplot

def TrainingTab():
    trainingtab = [
        html.H1("Benchmarking Time Series Graph using models"),
        html.H2("Make your own time series graph"),
        html.P("In an attempt to make this project more interactive, we are going to allow users to select the training data's date and variables they wish to use"),
        html.P("The chosen time period will be used as the training period for all 3 models. The training variables selected will be used in the ADL and RNN model"),
        dcc.Graph(id='time-series-graph', figure =mainplot(), className="graphBorder"),
        html.H4("Select training time period (Years)"),
        html.Div([slider()], className = "box"),
        html.Div([dropdown()],className = "box"),
        html.Strong(id = 'lag-caller'),
        html.P(),
        html.Button('Train the model!', id='train-model')
    ]
    return trainingtab
