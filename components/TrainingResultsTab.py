import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from components.dropdown_quarter import dropdown
from components.dropdown_year import dropdown_year
from data import mainplot

def TrainingResultsTab():
    training = [
        html.Div(id='training',
                 children = [
                     html.H2("Benchmarking Time Series Graph using models", style={'text-align': 'center', 'color': "white"}),
                     html.H3("Make your own time series graph", style={'text-align': 'center', 'color': "white"}),
                     html.P("In an attempt to make this project more interactive, we are going to allow users to select the training data's date.", style={'text-align': 'center', 'color': "white"}),
                     html.P("The chosen time period will be used as the training data for our Auto Regressive (AR), ADL, and Regression Forest (RF) model.", style={'text-align': 'center', 'color': "white"}),
                     dcc.Graph(id='time-series-graph', figure =mainplot(), className="graphBorder"),
                     html.H5("Select training time period"),
                     html.Div([
                         html.H6("Select a year"),
                         html.Div([dropdown_year()],className="dropdown-container"),
                         html.P(),
                         html.H6("Select a quarter"),
                         html.Div([dropdown()], className="dropdown-container")
                         ]), 
                     html.Strong(id = 'lag-caller'),
                     html.P(),
                     html.Button('Train the model!', id='train-model'),
                     html.Div(id='evaluation-results')
                     ])
                ]
    return training

