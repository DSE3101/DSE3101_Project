import dash
import numpy as np
import pandas as pd 
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
# from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from components.ARTab import ARTab
from components.TrainingTab import TrainingTab
from components.ADLTab import ADLTab
from components.MLTab import MLTab
from components.EvalTab import EvalTab
from data import mainplot


routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)
routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')
app = dash.Dash(__name__, external_stylesheets= [dbc.themes.SIMPLEX])

app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Model Training', className = "tab", children= TrainingTab()),
        dcc.Tab(label='AR', className="tab", children= ARTab()),
        dcc.Tab(label='ADL',className="tab" ,children=ADLTab()),
        dcc.Tab(label='ML', className="tab", children=MLTab()),
        dcc.Tab(label='Evaluation', className="tab", children=EvalTab())
    ]),
    html.Div(id='tabs-content')
])
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('date-slider', 'value'), Input('quarter-dropdown', 'value')]
)
def update_graph(slider_value, quarter_value):
    # Convert the slider value and quarter value to a date
    selected_year = date_range_yearly[slider_value].year
    selected_date_str = f"{selected_year}{quarter_value}"
    selected_date = pd.Period(selected_date_str, freq='Q').to_timestamp(how = 'end')
    filtered_data = routput[(routput['DATE'] <= selected_date)]
    
    # Create the figure
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=routput['DATE'], y=routput['ROUTPUT24Q1'], mode='lines', name='All data'))
    figure.add_trace(go.Scatter(x=filtered_data['DATE'], y=filtered_data['ROUTPUT24Q1'], mode='lines', name='Training data', line=dict(color='red', width=2)))
    
    # Update layout
    figure.update_layout(title='Time Series Data', xaxis_title='Date', yaxis_title='Value')
    figure.update_layout(margin=dict(l=20, r=20, t=30, b=20))  # left, right, top, bottom margin in pixels
    
    return figure

@app.callback(
    Output('lag-caller', 'children'),
    [Input('date-slider', 'value'), Input('quarter-dropdown', 'value')]
    )
def update_output(value, quarter_value):
    safe_value = min(value, len(date_range_yearly) - 1)
    selected_year = date_range_yearly[safe_value].year
    
    start_year = 1947
    start_quarter = 'Q1'
    
    selected_quarter_int = int(quarter_value.replace('Q', ''))
    start_quarter_int = int(start_quarter.replace('Q', ''))
    
    number_of_lags = (selected_year - start_year) * 4 + (selected_quarter_int - start_quarter_int)
    
    return f'Number of lags from Q1 1947 to {quarter_value} {selected_year}: {number_of_lags}'

@app.callback(
    Output('ar-plot', 'figure'),
              [Input('quarter-dropdown', 'value'), Input('date-slider', 'value')]
              )

def ARmodel(value, quarter_value):
    selected_year = date_range_yearly[int(value)].year
    start_year = 1947
    start_quarter = 'Q1'
    selected_quarter_int = int(quarter_value.replace('Q', ''))
    start_quarter_int = int(start_quarter.replace('Q', ''))
    num_lags = (selected_year - start_year) * 4 + (selected_quarter_int - start_quarter_int)
    
    period_t = "ROUTPUT" + selected_year%100 + "Q" + quarter_value
    real_time_data = routput[period_t].dropna()
    real_time_model = AutoReg(real_time_data, lags=num_lags)
    real_time_model_fit = real_time_model.fit()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_time_data.index, y=real_time_data, mode='lines', name='Real Time Data'))
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
