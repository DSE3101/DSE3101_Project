import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Generate dummy data
np.random.seed(123)
date_range = pd.date_range(start="1960-01-01", end="2023-12-31", freq='QE')
values = np.cumsum(np.random.uniform(-10, 10, len(date_range)))
dummy_data = pd.DataFrame({'date': date_range, 'value': values})

# For the slider component
date_range_yearly = pd.date_range(start='1960-01-01', end='2023-12-31', freq='YS')
timestamps = date_range.view('int64') // 10**9
marks = {i: {'label': str(year.year)} for i, year in enumerate(date_range_yearly)}
slider = dcc.RangeSlider(id='date-slider',
                            min=0,
                            max=len(date_range_yearly)-1,
                            value=[0, len(date_range_yearly)-1],
                            marks={i: {'label': str(year.year)} for i, year in enumerate(date_range_yearly)},
                            step=1  # Assuming you want to step through each year
                            )

#Checkbox component
checkbox = dcc.Checklist(
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
                value=[]  # Default selected values
            )

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets= [dbc.themes.SIMPLEX])
    
app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Model Training', children=[
            html.H1("Benchmarking Time Series Graph using models"),
            html.H2("Make your own time series graph"),
            html.P("In an attempt to make this project more interactive, we are going to allow users to select the training data's date and variables they wish to use"),
            dcc.Graph(id='time-series-graph', className="graphBorder"),
            html.H4("Select training time period (Years)"),
            html.Div([slider], className = "box"),
            html.H4("Select training variables"),
            html.Div([checkbox], className = "box"),
            html.Button('Train the model!', id='train-model')
        ]),
        dcc.Tab(label='ARIMA', className="tab", children=[
            html.Strong("Test!"),
            html.P("Content for Real Time vs Vintage Data.")
        ]),
        dcc.Tab(label='AR',className="tab" ,children=[
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
