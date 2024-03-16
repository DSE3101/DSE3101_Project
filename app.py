import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from components.slider import slider
from components.checkbox import checkbox
from data import generate_data
import numpy as np
import pandas as pd 

date_range_yearly = pd.date_range(start='1960-01-01', end='2023-12-31', freq='YS')
app = dash.Dash(__name__, external_stylesheets= [dbc.themes.SIMPLEX])
dummy_data = generate_data()
    
app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Model Training', children=[
            html.H1("Benchmarking Time Series Graph using models"),
            html.H2("Make your own time series graph"),
            html.P("In an attempt to make this project more interactive, we are going to allow users to select the training data's date and variables they wish to use"),
            dcc.Graph(id='time-series-graph', className="graphBorder"),
            html.H4("Select training time period (Years)"),
            html.Div([slider()], className = "box"),
            html.H4("Select training variables"),
            html.Div([checkbox()], className = "box"),
            html.Button('Train the model!', id='train-model')
        ]),
        dcc.Tab(label='AR', className="tab", children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
        dcc.Tab(label='ADL',className="tab" ,children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
        dcc.Tab(label='ML', className="tab", children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
        dcc.Tab(label='Evaluation', className="tab", children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('date-slider', 'value')]
)
def update_graph(slider_range):
    # Convert slider values to dates
    start_idx, end_idx = slider_range
    start_date = date_range_yearly[start_idx]
    end_date = date_range_yearly[min(end_idx, len(date_range_yearly) - 1)]  # Ensure end_idx is within bounds

    # Filter data
    filtered_data = dummy_data[(dummy_data['date'] >= start_date) & (dummy_data['date'] <= end_date)]
    
    # Create the figure
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dummy_data['date'], y=dummy_data['value'], mode='lines', name='All data'))
    figure.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['value'], mode='lines', name='Training data', line=dict(color='red', width=2)))
    
    # Update layout
    figure.update_layout(title='Time Series Data', xaxis_title='Date', yaxis_title='Value')
    figure.update_layout(margin=dict(l=20, r=20, t=30, b=20))  # left, right, top, bottom margin in pixels
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
