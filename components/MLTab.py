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


def MLTab():
    
    MLTab = html.Div([
        html.H3("Machine Learning Analysis"),
        html.P("In this section, we will use your selected training time period, coupled with our selected variables, to run a regression forest."),
        html.H4("Feature Importance"),
    ])

    return MLTab