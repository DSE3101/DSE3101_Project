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

def model_name(model_name, background_color):
        return html.Span(model_name, style={
            'background-color': background_color,
            'color': 'black',
            'padding': '8px 12px',
            'margin-right': '5px',
            'border-radius': '10px',
            'font-weight': 'bold',
            'text-align': 'center',
        })
armodel = model_name("AR Model", '#FF7F7F')
adlmodel = model_name("ADL Model", '#FDFD96') 
rfmodel = model_name("RF Model", '#90EE90')

def TrainingResultsTab():
    training = [
        html.Div(id='training',
                 children = [
                     html.H2("Can You Predict The Economy?", style={'text-align': 'center', 'color': "lightblue", 'padding': '10px'}),
                     html.H3("Benchmarking Time Series Forecasts Using Models ", style={'text-align': 'center', 'color': "white"}),
                     html.P(f"In an attempt to make this project more interactive, we are going to allow users to select the training data's date."
                            f" The chosen time period will be used as the training data for our ", style={'text-align': 'center', 'color': "white"}),
                     html.Div(children=[
                         armodel, adlmodel, rfmodel,
                     ], style={'text-align': 'center'}),
                     html.P(""),
                     dcc.Graph(id='time-series-graph', figure =mainplot(), className="graphBorder"),
                     html.H5("Select training time period"),
                     html.Div([
                         html.H6("Select a year"),
                         html.Div([dropdown_year()],className="dropdown-container"),
                         html.P(""),
                         html.H6("Select a quarter"),
                         html.Div([dropdown()], className="dropdown-container")
                         ]), 
                     html.Strong(id = 'lag-caller'),
                     html.P(),
                     html.Button('Train the model!', id='train-model'),
                    html.B("         Scroll down to see the results!"),

                     html.Div(id='evaluation-results'),
                     ])
                ]
    return training

