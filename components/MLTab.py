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
import pandas as pd
import pickle
from NewRandomForest import *
from app import app

selected_variables_importance_dict, rmsfe, real_time_plot, latest_plot, y_pred = random_forest(year, quarter)

def MLTab():
    MLTab = html.Div([
        html.H3("Machine Learning Analysis"),
        html.P("In this section, we will use your selected training time period, coupled with our selected variables, to run a regression forest."),
        html.H4("Feature Importance"),
        html.Ul([
        html.Li(f"{var}: {imp}") for var, imp in selected_variables_importance_dict.items()
        ])

    ])

    return MLTab


#Server
#Access data used
@app.callback(
    Output('year-quarter', 'data'),
    [Input('dropdown-year', 'value'), Input('dropdown-quarter', 'value')]
)
def update_shared_data(year_value, quarter_value):
    data = {
        'year': str(year_value),
        'quarter': str(quarter_value)
    }
    return data

#Run ML
@app.callback(
    Output('ML-content', 'children'),  
    [Input('year-quarter', 'data')])

def ml_function(year_quarter):
     year = year_quarter['year']
     quarter = year_quarter['quarter']
     selected_variables_importance_dict, rmsfe, real_time_plot, latest_plot, y_pred = random_forest(year, quarter)