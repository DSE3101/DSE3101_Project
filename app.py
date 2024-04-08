import dash
import numpy as np
import pandas as pd 
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from components.ARTab import ARTab
from components.TrainingResultsTab import TrainingResultsTab
from components.ADLTab import ADLTab
from components.MLTab import *
from data import mainplot
import pickle
from GetData import get_data
from sklearn.metrics import mean_squared_error
from AR_Model import AR_MODEL
from RandomForest import *

routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)
routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
routput = routput[routput['DATE'].dt.year >= 1965]

date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')
app = dash.Dash(__name__, external_stylesheets= [dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='model-training', children=[
        dcc.Tab(label='Model Training', value='model-training', className="tab", children=TrainingResultsTab()),
        dcc.Tab(label='AR', value='ar', className="tab", children=ARTab()),
        dcc.Tab(label='ADL', value='adl', className="tab", children=ADLTab()),
        dcc.Tab(label='ML', value='ml', className="tab", children=MLTab()),
]),
    html.Div(id='tabs-content'),
    dcc.Store(id = 'year-quarter')
])

#Slider elements will update the graph
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('dropdown-year', 'value'), Input('dropdown-quarter', 'value')]
)
def update_graph(year_value, quarter_value):
    # Convert the slider value and quarter value to a date
    selected_date_str = f"{year_value}{quarter_value}"
    selected_date = pd.Period(selected_date_str, freq='Q').to_timestamp(how='end')
    filtered_data = routput[(routput['DATE'] <= selected_date)]
    
    # Create the figure
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=routput['DATE'], y=routput['ROUTPUT24Q1'], mode='lines', name='All data'))
    figure.add_trace(go.Scatter(x=filtered_data['DATE'], y=filtered_data['ROUTPUT24Q1'], mode='lines', name='Training data', line=dict(color='red', width=2)))
    
    # Update layout
    figure.update_layout(title='Time Series Data', xaxis_title='Date', yaxis_title='Value')
    figure.update_layout(margin=dict(l=20, r=20, t=30, b=20))  # left, right, top, bottom margin in pixels
    
    return figure

#Data taken from slider and quarter will be shown as a reactive statement in trainingtab
@app.callback(
    Output('lag-caller', 'children'),
    [Input('dropdown-year', 'value'), Input('dropdown-quarter', 'value')]
    )

def update_output(year_value, quarter_value):
    return f'Your training data will be from 1947 Q1 to {year_value} {quarter_value}'

#Data taken from training tab will be called to AR, ADL and ML
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

#Train model button
@app.callback(
    [Output('evaluation-results', 'children'),  
     Output('evaluation-results', 'style')],   
    [Input('train-model', 'n_clicks')],
    [State('year-quarter', 'data')] 
)
def update_evaluation_results_and_show(n_clicks, year_quarter_data):
    if n_clicks is None or year_quarter_data is None:
        return [], {'display': 'none'}

    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")

    ar_model_results = AR_MODEL(year, quarter)
    random_forest_results = random_forest(year, quarter)
    
    evaluation = html.Div([
        html.H3("Evaluating our models", style={'text-align': 'center'}),
        html.Div([
            html.H5("AR Model", className="model-header"),
            html.Div([
                # Column for Real-Time Data
                html.Div([
                    html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                    html.Div([html.B("Real Time Data RMSE:"),
                              html.P(f"{round(ar_model_results[2],3)}")],
                             className="rmse-value")
                ], className="graph-container"),
                # Column for Vintage Data 
                html.Div([
                    html.Img(src='assets/ar_vintage_plot.png', className="graph-image"),
                    html.Div([html.B("Real Time Data RMSE:"),
                              html.P(f"{round(ar_model_results[2],3)}")],
                             className="rmse-value")
                ], className="graph-container"),
            ], className="model-container"),
            #ADL Model
            html.H5("ADL Model", className="model-header"),
            html.Div([
                # Column for Real-Time Data
                html.Div([
                    html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                    html.Div([html.B("Real Time Data RMSE:"),
                              html.P(f"{round(ar_model_results[2],3)}")],
                             className="rmse-value")
                ], className="graph-container"),
                # Column for Vintage Data 
                html.Div([
                    html.Img(src='assets/ar_vintage_plot.png', className="graph-image"),
                    html.Div([html.B("Real Time Data RMSE:"),
                              html.P(f"{round(ar_model_results[2],3)}")],
                             className="rmse-value")
                ], className="graph-container"),
            ], className="model-container"),
            #RF
            html.H5("Random Forest Model", className="model-header"),
            # Column for Real-Time Data
            html.Div([
                html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                html.Div([html.B("Real Time Data RMSE:"),
                            html.P(f"{round(random_forest_results[1],3)}")],
                            className="rmse-value")
            ], className="graph-container"),
            # Column for Vintage Data 
            html.Div([
                html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                html.Div([html.B("Real Time Data RMSE:"),
                          html.P(f"{round(random_forest_results[4],3)}")],
                         className="rmse-value")
                ], className="graph-container"),
            ], className="model-container"),
        ], className="evaluation-container")
    style={'background-color': 'lightblue', 'padding': '20px', 'border-radius': '5px', 'margin': '20px','display': 'block'}
    return evaluation, {'display': 'block'}


if __name__ == '__main__':
    app.run_server(debug=True)
