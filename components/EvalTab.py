import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def EvalTab():
    EvalTab = [
        html.Strong("Test!"),
        html.P("Content for Real Time vs Vintage Data.")
        ]
    return EvalTab