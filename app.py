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
from dm import *


from components.ARTab import ARTab
from components.TrainingResultsTab import TrainingResultsTab
from components.ADLTab import ADLTab
from components.MLTab import *
from data import mainplot #Main graph on landing
from GetData import get_data
from NewRandomForest import *
from NewARModel import *


routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)
routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
routput = routput[routput['DATE'].dt.year >= 1965]
date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.DARKLY], suppress_callback_exceptions=True)
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Training and Explaining the Model", href="/model-training")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("AR", href="/ar"),
                dbc.DropdownMenuItem("ADL", href="/adl"),
                dbc.DropdownMenuItem("ML", href="/ml"),
            ],
            nav=False,
            in_navbar=True,
            label="How the models worked",
        ),
    ],
    brand="Economics Forecasting",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    dcc.Store(id='year-quarter'),
    dcc.Store(id='rf-results'),
    dcc.Store(id='ar-results'),
    dcc.Store(id='adl-results'),
])

### SERVER
#Navbar
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/model-training':
        return TrainingResultsTab()  # Replace with your function for Model Training
    elif pathname == '/ar':
        return ARTab()  # Replace with your function for AR
    elif pathname == '/adl':
        return ADLTab()  # Replace with your function for ADL
    elif pathname == '/ml':
        return MLTab()  # Replace with your function for ML
    else:
        # If the user tries to reach a different page, redirect them to the home page
        return TrainingResultsTab()

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

#Run AR
@app.callback(
    Output('ar-results', 'data'),
    [Input('train-model', 'n_clicks')],
    [State('year-quarter', 'data')]
)
def ar_results(n_clicks, year_quarter_data):
    if not year_quarter_data:
        return dash.no_update
    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")
    ar_real_time_optimal_lags, ar_real_time_rmsfe,ar_real_time_plot, ar_y_pred = AR_MODEL(year, quarter)
    ar_results = {
            'optimal_lags': ar_real_time_optimal_lags,
            'rmsfe': ar_real_time_rmsfe,
            'plot': ar_real_time_plot,
            'y_pred': ar_y_pred
        }
    return ar_results

#Run ADL

#Run RF
@app.callback(
    Output('rf-results', 'data'),
    [Input('train-model', 'n_clicks')],
    [State('year-quarter', 'data')]
)
def rf_results(n_clicks, year_quarter_data):
    if not year_quarter_data:
        return dash.no_update
    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")
    rf_selected_variables_importance_dict, rf_rmsfe, rf_real_time_plot, rf_y_pred = random_forest(year, quarter)    
    rf_results = {
            'selected_variables': rf_selected_variables_importance_dict,
            'rmsfe': rf_rmsfe,
            'plot': rf_real_time_plot,
            'y_pred': rf_y_pred
        }
    return rf_results

#Train model button
@app.callback(
    [Output('evaluation-results', 'children'),  
     Output('evaluation-results', 'style')],   
    [Input('ar-results', 'data'), 
     #Input('adl-results', 'data'), 
     Input('rf-results', 'data'),
     Input('year-quarter', 'data')]
)
def update_evaluation_results_and_show(ar_results, rf_results, year_quarter_data):
    if not ar_results or not rf_results:
        return [], {'display': 'none'}

    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")

    #get_data, will be used for dm_test
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year, quarter)

    #DM test
    #ar_adl_dm = DM(ar_h_realtime, adl_h_realtime, real_time_y, h = 8)
    rf_p_value = DM(ar_results['y_pred'], rf_results['y_pred'], latest_y_test, h = 8)[2]
    
    #DM explainer
    low_p_value ="Since the p-value is less than 0.05, it means that there is significant predictive capabilities between the two models."
    high_p_value = "Since the p-value of the DM test is more than 0.05, it means that both models have similar predictive capabilities. Thus, it is okay to use the real-time data to predict future values and vice-versa"

    evaluation = html.Div([
        html.Div([
            html.H3("Evaluating our models", style={'text-align': 'center', 'color': "black"}),
            html.P("Using the training data selected above, we will now use 3 different forecasting methods to forecast the next 12 quarters.", style={'text-align': 'center', 'color': "black"}),
            html.P("The three methods used will be the AR model, ADL model, and the Random Forest.", style={'text-align': 'center', 'color': "black"}),
            html.P("We will be using both the RMSFE and DM to evaluate the models and determine which is the most suitable given the training period.", style={'text-align': 'center', 'color': "black"}),
        ], className= "evaluation-container"),
    
    #RMSE Section
    html.Div([
        html.Div([
            html.H4("RMSFE Evaluation"),
            html.P("A lower RMSFE indicates that the model fits the historical closely, making it more accurate for future prediction."),
        # AR Model Container
            html.Div([
                html.H5("AR Model", className="model-header"),
                html.Img(src=ar_results['plot'], className="graph-image graphBorder"),
                html.P("AR Model RMSFE: ", style={'color': 'black'}),
                html.P(f"{round(ar_results['rmsfe'], 3)}", className="rmse-value"),
                # AR Model Write-up Section
                html.Div([
                    html.P(f"We have trained the AR model using your selection of training data of {year} Q{quarter}.", style={'color': 'black'}),
                    html.P(f" Using a rolling window average to train our model, our 8 step forecast has indicated that the RMSFE is {ar_results['rmsfe']}.", style={'color': 'black'}),
                ], className="write-up-container"),
            ], className="model-container"),
            
            # ADL Model Container
            html.Div([
                html.H5("ADL Model", className="model-header"),
                html.Img(src=rf_results['plot'], className="graph-image graphBorder"),
                html.P("ADL Model RMSFE: ", style={'color': 'black'}),
                html.P(f"{round(rf_results['rmsfe'], 3)}", className="rmse-value"),
                # ADL Model Write-up
                html.Div([
                    html.P(f"We have trained the ADL model using your selection of training data of {year} Q{quarter}.", style={'color': 'black'}),
                    html.P(f" Using a rolling window average to train our model, our 8 step forecast has indicated that the RMSFE is {rf_results['rmsfe']}.", style={'color': 'black'}),
                ], className="write-up-container"),
            ], className="model-container"),
            
            # RF Model Container
            html.Div([
                html.H5("RF Model", className="model-header"),
                html.Img(src=rf_results['plot'], className="graph-image graphBorder"),
                html.P("RF Model RMSFE: ", style={'color': 'black'}),
                html.P(f"{round(rf_results['rmsfe'], 3)}", className="rmse-value"),
                # RF Model Write-up Section
                html.Div([
                    html.P(f"We have trained the RF model using your selection of training data of {year} Q{quarter}.", style={'color': 'black'}),
                    html.P(f" Using a rolling window average to train our model, our 8 step forecast has indicated that the RMSFE is {rf_results['rmsfe']}.", style={'color': 'black'}),
                    html.P(f"{rf_p_value}", style={'color': 'black'})
                    ], className="write-up-container"),
                ], className="model-container"),
            ], className="model-split-container", style={'display': 'flex', 'justify-content': 'space-around'})
        ],className="evaluation-container", style={'background-color': 'lightblue', 'padding': '20px', 'border-radius': '5px', 'margin': '20px', 'display': 'flex', 'flex-direction': 'column', 'gap': '20px'}),
    
    #DM Section    
    html.Div([
        html.Div([
            html.H4("Diebold-Mariano (DM) Test"),
            html.P("A DM test is especially useful when it comes to comparing the performance between two models")
            ])],
             className="evaluation-container", style={'background-color': 'lightblue', 'padding': '20px', 'border-radius': '5px', 'margin': '20px', 'display': 'flex', 'flex-direction': 'column', 'gap': '20px'})
    ])

    return evaluation, {'display': 'block'}


if __name__ == '__main__':
    app.run_server(debug=True)
