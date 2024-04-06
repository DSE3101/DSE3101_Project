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

def TrainingResultsTab():
    training = [
        html.Div(id='training', children = [
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
        html.Button('Train the model!', id='train-model'),
        ])
    ]

    evaluation = [
        html.Div(id='evaluation-results', style={'display': 'none'}, children=[
            html.H3("Evaluating our models"),
            html.P("We will use 2 tests to determine which model is the most appropriate"),
            html.P("The tests are RMSE and Diebold-Mariano Test"),
            html.H4("RMSE"),
            html.P("RMSE is an extremely simple and easy to implement test"),
            ##Insert 3 graphs, which are 3 fancharts, and with the test data
            html.H5("AR Model"),
            #Graph 1
            html.P("The AR model serves as our baseline model. Based on econometric theory, it should be the worst performing in terms of RMSE values"),
            #html.P("Running the AR model described above, the RMSE value is" + rmse_ar),
            html.P("The value is quite high, so we will see how our next model fare"),
            html.P("Based on our models, the model with the lowest RMSE is xxx"),
            html.H5("ADL Model"),
            #Graph 2
            #html.P("The ADL model will return us a RMSE value of " + rmse_adl), #Text will be on the right
            html.H5("Regression Forest Model"),
            #Graph 3
            #html.P("Last but not least, running our own regression model returns us the RMSE value of " + rmse_forest),
            
            html.P("But using the RMSE has its own issues,"),
            html.P("For instance, RMSE are extremely sensitive to outliers, and is not as statistically sound as our other test."),
            html.P("This is where our next test comes in"),

            html.H4("Diebold-Mariano (DM) Test"),
            html.P("We will now run the DM test between the ADL and regression forest model."), 
            html.P("The reason why we are not running this on the AR model is because we have already established that the AR model is simply a benchmark model.")
        ])
    ]
    return training + evaluation

