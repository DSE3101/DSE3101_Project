import sys
print(sys.path)


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Generate dummy data
np.random.seed(123)
date_range = pd.date_range(start="1960-01-01", end="2023-12-31", freq='Q')
values = np.cumsum(np.random.uniform(-10, 10, len(date_range)))
dummy_data = pd.DataFrame({'date': date_range, 'value': values})

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Model Training', children=[
            html.H1("Benchmarking Time Series Graph using models"),
            html.H2("Make your own time series graph"),
            html.P("In an attempt to make this project more interactive, we are going to allow users to select the training data's date and variables they wish to use"),
            dcc.Graph(id='time-series-graph'),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=date_range.min(),
                max_date_allowed=date_range.max(),
                start_date='1980-01-01',
                end_date='2020-01-01'
            ),
            dcc.Checklist(
                id='checkbox-menu',
                options=[
                    {'label': 'M1', 'value': 'opt2'},
                    {'label': 'Investment', 'value': 'opt3'},
                    {'label': 'Total Reserves', 'value': 'opt4'},
                    {'label': 'Nonborrowed Reserves', 'value': 'opt5'},
                    {'label': 'Nonborrowed Reserves + Extended Credit', 'value': 'opt6'},
                    {'label': 'Monetary Base', 'value': 'opt7'},
                    {'label': 'Civilian Unemployed Rate', 'value': 'opt8'},
                    {'label': 'CPI vs Chain-weighted Price Index', 'value': 'opt9'},
                    {'label': '3-month T Bill Rate', 'value': 'opt10'},
                    {'label': '10-year T-bond Rate', 'value': 'opt11'}
                ],
                value=['opt2', 'opt3']  # Default selected values
            ),
            html.Button('Train the model!', id='train-model')
        ]),
        dcc.Tab(label='ARIMA', children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
        # Add other tabs as needed
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_data = dummy_data[(dummy_data['date'] >= start_date) & (dummy_data['date'] <= end_date)]
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dummy_data['date'], y=dummy_data['value'], mode='lines', name='All data'))
    figure.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['value'], mode='lines', name='Selected data', line=dict(color='red', width=2)))
    
    figure.update_layout(title='Dummy Time Series Graph', xaxis_title='Date', yaxis_title='Value')
    
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
