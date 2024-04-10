import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def ARTab():
    ar_model_content = html.Div([
        html.H3("AR MODEL"),
        html.P("We will be using the 2012 Q2 prediction to explain our models"),

        html.H4("How does the model work"),
        html.P("Our autoregressive (AR) model uses past values of GDP to forecast GDP. As such, forecasted GDP values are regressed against its own lagged values."),

        html.H4("Finding the optimal number of lags"),
        html.P("Although our team intended to run an 8 step forecast, we realized that the most optimal way to get the correct number of lags to use was to run a one step forecast, using the minimum AIC and BIC values. This will be the identified model to predict the next one step forecast."),
        html.Img(src="assets/ar_tab/ar_lag.png"),

        html.H4("Expanding window"),
        html.P("We repeat the above-mentioned method will be iterated 8 times. Giving and to get a model to predict y+1 using the AIC and BIC. We then repeat this step 8 times, using the predicted y+1/y+2/… values all the way until y+7 to predict the y+8 forecast. This will generate 8 prediction values as shown."),
        html.Img(src="assets/ar_tab/forecast.png"),

        html.H4("Calculating RMSFE"),
        html.P("We then calculated the RMSFE using the root mean squared error between the true y values and the predicted y values. In the case of 2012 Q2, the returned RMSFE value was 0.00534.")
        ],style={'text-align': 'justify'})

    return ar_model_content