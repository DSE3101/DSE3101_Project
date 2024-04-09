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
import pickle
from sklearn.metrics import mean_squared_error
import base64
from io import BytesIO


from components.ARTab import ARTab
from components.TrainingResultsTab import TrainingResultsTab
from components.ADLTab import ADLTab
from components.MLTab import *
from data import mainplot #Main graph on landing
from GetData import get_data
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
        dcc.Tab(label='AR', value='ar', className="tab", children=ARTab()), #explainer, AC plots, AIC
        dcc.Tab(label='ADL', value='adl', className="tab", children=ADLTab()),
        dcc.Tab(label='ML', value='ml', className="tab", children=MLTab()),
]),
    html.Div(id='tabs-content'),
    dcc.Store(id = 'year-quarter')
])

### SERVER
#Dropdown elements will update the graph
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
    if n_clicks is None:
        return [], {'display': 'none'}

    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")

    ar_model_results = AR_MODEL(year, quarter)
    random_forest_results = random_forest(year, quarter)
    
    evaluation = html.Div([
        html.H3("Evaluating our models", style={'text-align': 'center', 'color' :"black"}),
        # AR Model Container
    html.Div([
        html.H5("AR Model", className="model-header"),
        # Model Split Container for graphs and metrics
        html.Div([
            # Real Time Data Column
            html.Div([
                html.Img(src=ar_model_results[6], className="graph-image"),
                html.Div([
                    html.B("Real Time Data RMSFE: ", style={'color': 'black'}),
                    html.P(f"{round(ar_model_results[2], 3)}", className="rmse-value")
                ], className="rmse-box")
            ], className="graph-container"),

            # Vintage Data Column
            html.Div([
                html.Img(src=ar_model_results[7], className="graph-image"),
                html.Div([
                    html.B("Vintage Data RMSFE: ", style={'color': 'black'}),
                    html.P(f"{round(ar_model_results[5], 3)}", className="rmse-value")
                ], className="rmse-box")
            ], className="graph-container"),
        ], className="model-split-container"),
        
        # Write-up Section
        html.Div([
            html.P(f"Here is an insightful analysis based on the AR model results. This analysis provides an in-depth look at how the model performs with real-time and vintage data, highlighting the key takeaways and implications for future forecasting.", style={'color': 'black'}),
        ], className="write-up-container"),
    ], className="model-container"),

        # ADL Model Container (Repeat the structure as needed for other models)
        html.Div([
            html.H5("ADL Model", className="model-header"),
            html.Div([
                # Real Time Data Column
                html.Div([
                    html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                    html.Div([
                        html.B("Real Time Data RMSFE: ", style = {'color':'black'}),
                        html.P(f"{round(ar_model_results[2], 3)}", className="rmse-value")
                    ], className="rmse-box")
                ], className="graph-container"),

                # Vintage Data Column
                html.Div([
                    html.Img(src='assets/ar_vintage_plot.png', className="graph-image"),
                    html.Div([
                        html.B("Vintage Data RMSFE: ", style = {'color':'black'}),
                        html.P(f"{round(ar_model_results[5], 3)}", className="rmse-value")
                    ], className="rmse-box")
                ], className="graph-container"),
            ], className="model-split-container"),

            # Write-up Section
        html.Div([
            html.P(f"Here is an insightful analysis based on the AR model results. This analysis provides an in-depth look at how the model performs with real-time and vintage data, highlighting the key takeaways and implications for future forecasting.", style={'color': 'black'}),
        ], className="write-up-container"),
    ], className="model-container"),
        
        # RF Model Container (Repeat the structure as needed for other models)
        html.Div([
            html.H5("RF Model", className="model-header"),
            html.Div([
                # Real Time Data Column
                html.Div([
                    html.Img(src='assets/ar_real_time_plot.png', className="graph-image"),
                    html.Div([
                        html.B("Real Time Data RMSFE: ", style = {'color':'black'}),
                        html.P(f"{round(random_forest_results[1], 3)}", className="rmse-value")
                    ], className="rmse-box")
                ], className="graph-container"),

                # Vintage Data Column
                html.Div([
                    html.Img(src='assets/ar_vintage_plot.png', className="graph-image"),
                    html.Div([
                        html.B("Vintage Data RMSFE: ", style = {'color':'black'}),
                        html.P(f"{round(random_forest_results[4], 3)}", className="rmse-value")
                    ], className="rmse-box")
                ], className="graph-container"),
            ], className="model-split-container"),
            
            # Write-up Section
        html.Div([
            html.P(f"Here is an insightful analysis based on the AR model results. This analysis provides an in-depth look at how the model performs with real-time and vintage data, highlighting the key takeaways and implications for future forecasting.", style={'color': 'black'}),
        ], className="write-up-container"),
    ], className="model-container"),
        #Final part
    ], className="evaluation-container", style={'background-color': 'lightblue', 'padding': '20px', 'border-radius': '5px', 'margin': '20px', 'display': 'flex', 'flex-direction': 'column', 'gap': '20px'})

    return evaluation, {'display': 'block'}


if __name__ == '__main__':
    app.run_server(debug=False)
