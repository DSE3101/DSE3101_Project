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
from dash import dash_table



from components.ARTab import *
from components.TrainingResultsTab import TrainingResultsTab
from components.ADLTab import ADLTab
from components.MLTab import *
from data import mainplot #Main graph on landing
from GetData import get_data
from RandomForestFixed import *   
from ARModelFixed import *
from NewADLModel import *


routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)
routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
routput = routput[routput['DATE'].dt.year >= 1965]
date_range_yearly = pd.date_range(start='1947-01-01', end='2023-12-31', freq='YS')

app = dash.Dash(__name__, external_stylesheets= [dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Training and Evaluating the Model", href="/model-training")),
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
        return TrainingResultsTab() 
    elif pathname == '/ar':
        return ARTab() 
    elif pathname == '/adl':
        return ADLTab()
    elif pathname == '/ml':
        return MLTab() 
    else:
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
    if n_clicks is None or not year_quarter_data:
        return dash.no_update
    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")
    h_step_forecast, h_step_lag, abs_error, real_time_plot = AR_MODEL(year, quarter)
    ar_results = {
            'lag': h_step_lag,
            'abs_error' : abs_error,
            'rmsfe': 0.1,
            'plot': real_time_plot,
            'y_pred': h_step_forecast
        }
    return ar_results

#Run ADL
@app.callback(
    Output('adl-results', 'data'),
    [Input('train-model', 'n_clicks')],
    [State('year-quarter', 'data')]
)
def adl_results(n_clicks, year_quarter_data):
    if n_clicks is None or not year_quarter_data:
        return dash.no_update
    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")
    lags_dict, y_pred, rmsfe, adl_plot = ADL_MODEL(year, quarter)
    adl_results = {
            'optimal_lags': lags_dict,
            'rmsfe': rmsfe,
            'plot': adl_plot,
            'y_pred': y_pred
        }
    return adl_results

#Run RF
@app.callback(
    Output('rf-results', 'data'),
    [Input('train-model', 'n_clicks')],
    [State('year-quarter', 'data')]
)
def rf_results(n_clicks, year_quarter_data):
    if n_clicks is None or not year_quarter_data:
        return dash.no_update
    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")
    rf_rmsfe, rf_real_time_plot, rf_y_pred = random_forest(year, quarter)    
    rf_results = {
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
     Input('adl-results', 'data'), 
     Input('rf-results', 'data'),
     Input('year-quarter', 'data')]
)
def update_evaluation_results_and_show(ar_results, adl_results, rf_results, year_quarter_data):
    if not ar_results or not rf_results:
        return [], {'display': 'none'}
    
    def model_name(model_name, background_color):
        return html.Span(model_name, style={
            'background-color': background_color,
            'color': 'black',
            'padding': '8px 12px',
            'margin-right': '5px',
            'border-radius': '10px',
            'font-weight': 'bold',
            'text-align': 'center',
        })
    armodel = model_name("AR Model", '#FF7F7F')
    adlmodel = model_name("ADL Model", '#FDFD96') 
    rfmodel = model_name("RF Model", '#90EE90')

    year = year_quarter_data['year']
    quarter = year_quarter_data['quarter'].replace("Q", "")

    #get_data, will be used for dm_test
    real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter = get_data(year, quarter)

    #DM test
    ar_adl_dm = round(DM(ar_results['y_pred'], adl_results['y_pred'], latest_y_test, h = 8)[2],3)
    ar_rf_dm = round(DM(ar_results['y_pred'], rf_results['y_pred'], latest_y_test, h =8)[2],3)
    
    #DM explainer
    low_p_value ="Since the p-value is less than 0.05, it means that there is significant predictive capabilities between the AR model and this model."
    high_p_value = "Since the p-value of the DM test is more than 0.05, there is no significant predictive capabilities of the AR model and this model."

    adl_p_value_explanation = low_p_value if ar_adl_dm < 0.05 else high_p_value
    rf_p_value_explanation = low_p_value if ar_rf_dm < 0.05 else high_p_value

    evaluation = html.Div([
        #Intro
        html.Div([
            html.H3("Evaluating our models", style={'text-align': 'center', 'color': "black"}),
            html.P("Using the training data selected above, we will now use 3 different forecasting methods to forecast the next 8 quarters."),
            html.P(["The three methods used will be the " , armodel , ", " , adlmodel , ", and ",  rfmodel ,"."]),
            html.P("We will be using both the RMSFE and DM to evaluate the models and determine which is the most suitable given the training period."),
        ], className= "evaluation-container"),
    
        #RMSE Section
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H4("RMSFE Evaluation"),
                    html.P("A lower RMSFE indicates that the model fits the historical data closely, making it more accurate for future predictions."),
                ]), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.P([armodel]),
                    html.P(""),
                    html.Img(src=ar_results['plot'], className="graph-image", style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.B(f"AR Model RMSFE: {round(ar_results['rmsfe'], 3)}",style={'text-align': 'center', 'color': "black"}),
                    html.P(f"We have trained the AR Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, our 8 step forecast has indicated that the RMSFE is {round(ar_results['rmsfe'], 3)}."),
                ]), width=4),
                dbc.Col(html.Div([
                    html.P([adlmodel]),
                    html.Img(src=adl_results['plot'], className="graph-image",style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.B(f"ADL Model RMSFE: {round(adl_results['rmsfe'], 3)}",style={'text-align': 'center', 'color': "black"}),
                    html.P(f"We have trained the ADL Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, our 8 step forecast has indicated that the RMSFE is {round(adl_results['rmsfe'],3)}."),
                ]), width=4),
                dbc.Col(html.Div([
                    html.P([rfmodel]),
                    html.Img(src=rf_results['plot'], className="graph-image",style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.B(f"RF Model RMSFE: {round(rf_results['rmsfe'], 3)}",style={'text-align': 'center', 'color': "black"}),
                    html.P(f"We have trained the RF Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, our 8 step forecast has indicated that the RMSFE is {round(rf_results['rmsfe'],3)}."),
                    ]), width=4),
                ]),
            ], className= 'evaluation-container'),

    #DM Section    
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H4("Diebold-Mariano (DM) Test"),
                    html.P(f"A DM test is especially useful when it comes to comparing the performance between two models. "
                        f"We will be using the AR Model as our benchmark to determine if our ADL Model and RF Model will perform similarly in terms of predictive capability."),
                ]), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("ADL Model"),
                    html.B(f"ADL Model p-value: {ar_adl_dm}", style={'text-align': 'center', 'color': "black"}),
                    html.P(adl_p_value_explanation),
                ]), width=6),
                dbc.Col(html.Div([
                    html.H5("RF Model"),
                    html.B(f"RF Model p-value: {ar_rf_dm}", style={'text-align': 'center', 'color': "black"}),
                    html.P(rf_p_value_explanation),
                ]), width=6),
            ]),
        ], className="evaluation-container")
    ])

    return evaluation, {'display': 'block'}


if __name__ == '__main__':
    app.run_server(debug=True)
