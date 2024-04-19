import dash
import pandas as pd 
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import os

#Importing components and tests
from Backend.dm import *
from Frontend.components.ARTab import *
from Frontend.components.TrainingResultsTab import *
from Frontend.components.ADLTab import *
from Frontend.components.MLTab import *
from Backend.GetData import *
from Backend.model_AR import *
from Backend.model_ADL import *
from Backend.model_RF import *


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
    year = int(year_value)
    quarter = int(quarter_value.strip('Q'))
    start_year = 1947 
    start_quarter = 1
    selected_date_str = f"{year_value}{quarter_value}"
    selected_period_index = pd.Period(selected_date_str, freq='Q')
    start_date_str = f"{start_year}Q{start_quarter}"
    start_period_index = pd.Period(start_date_str, freq='Q')

    # Filter data between the start and selected periods
    filtered_data = routput[(routput['DATE'] >= start_period_index.start_time) & (routput['DATE'] <= selected_period_index.end_time)]

    
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
    forecasts, optimal_lags, rmse, plot = AR_MODEL(year, quarter)
    ar_results = {
            'lag': optimal_lags,
            'rmsfe': rmse,
            'plot': plot,
            'y_pred': forecasts
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
    _, optimal_lags, _, _ = AR_MODEL(year, quarter)

    plot, forecasts, rmsfe = ADL_MODEL(year, quarter, optimal_lags)
    adl_results = {
        'rmsfe': rmsfe,
        'y_pred': forecasts,
        'plot': plot,
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
    rmse, real_time_plot, y_pred = RANDOM_FOREST(year, quarter)    
    rf_results = {
            'rmsfe': rmse,
            'plot': real_time_plot,
            'y_pred': y_pred
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
    ar_adl_dm = round(DM(ar_results['y_pred'], adl_results['y_pred'], latest_y_test, h = 8)[0],3)
    ar_rf_dm = round(DM(ar_results['y_pred'], rf_results['y_pred'], latest_y_test, h =8)[0],3)
    
    #DM explainer
    low_t_value ="Since the t-value is less than -1.96, it indicates that the AR model has statistically significant better predictive performance compared to this model at the 5% significance level."
    high_t_value ="Since the t-value is more than 1.96, it indicates that this model has statistically significant better predictive performance compared to the AR model at the 5% significance level."
    normal_t_value="Since the t-value is more than -1.96 and less than 1.96, it indicates that both models have no differences in predictive capabilities at the 5% significance level"

    if ar_adl_dm < -1.96:
        adl_t_value_explanation = low_t_value
    elif ar_adl_dm > 1.96:
        adl_t_value_explanation = high_t_value
    else:
        adl_t_value_explanation = normal_t_value 
        
           
    if ar_rf_dm < -1.96:
        rf_t_value_explanation = low_t_value
    elif ar_rf_dm > 1.96:
        rf_t_value_explanation = high_t_value
    else:
        rf_t_value_explanation = normal_t_value 
    
    #AR Table
    headers = ["Model Number", "RMSE"]

    model_numbers =[f"{i}" for i in range(len(ar_results['rmsfe']))]
    ar_rmses = ar_results['rmsfe']
    ar_table = html.Table(
        [html.Tr([html.Th(header) for header in headers])] +
        [html.Tr([html.Td(model_numbers[i]), html.Td(round(ar_rmses[i],4))]) for i in range(len(model_numbers))]
    )
    
    adl_rmses = adl_results['rmsfe']
    adl_table = html.Table(
        [html.Tr([html.Th(header) for header in headers])] +
        [html.Tr([html.Td(model_numbers[i]), html.Td(round(adl_rmses[i],4))]) for i in range(len(model_numbers))]
    )
    
    rf_rmses = rf_results['rmsfe']
    rf_table = html.Table(
        [html.Tr([html.Th(header) for header in headers])] +
        [html.Tr([html.Td(model_numbers[i]), html.Td(round(rf_rmses[i],4))]) for i in range(len(model_numbers))]
    )

    evaluation = html.Div([
        #Intro
        html.Div([
            html.H3("Evaluating our models", style={'text-align': 'center', 'color': "black"}),
            html.P("Using the training data selected above, we will now use 3 different forecasting methods to forecast the next 8 quarters."),
            html.P(["The three methods used will be the " , armodel , ", " , adlmodel , ", and ",  rfmodel ,"."]),
            html.P("We will be using both the RMSE and DM to evaluate the models and determine which is the most suitable given the training period."),
        ], className= "evaluation-container"),
    
        #RMSE Section
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H4("RMSE Evaluation"),
                    html.P("A lower RMSE indicates that the model fits the historical data closely, making it more accurate for future predictions."),
                ]), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.P([armodel]),
                    html.P(""),
                    html.Img(src=ar_results['plot'], className="graph-image", style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.Div([ar_table],style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.P(f"We have trained the AR Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, we've developed 8 different models to forecast 8 different points in our forecast."),
                ]), width=4),
                dbc.Col(html.Div([
                    html.P([adlmodel]),
                    html.Img(src=adl_results['plot'], className="graph-image",style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.Div([adl_table],style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.P(f"We have trained the ADL Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, we've developed 8 different models to forecast 8 different points in our forecast."),
                ]), width=4),
                dbc.Col(html.Div([
                    html.P([rfmodel]),
                    html.Img(src=rf_results['plot'], className="graph-image",style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.Div([rf_table],style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    html.P(f"We have trained the RF Model using your selection of training data of {year} Q{quarter}. Using a fixed window approach to train and backtest our model, we've developed 8 different models to forecast 8 different points in our forecast."),
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
                    html.P(adl_t_value_explanation),
                ]), width=6),
                dbc.Col(html.Div([
                    html.H5("RF Model"),
                    html.B(f"RF Model p-value: {ar_rf_dm}", style={'text-align': 'center', 'color': "black"}),
                    html.P(rf_t_value_explanation),
                ]), width=6),
            ]),
        ], className="evaluation-container")
    ])

    return evaluation, {'display': 'block'}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run_server(debug=True, host='0.0.0.0', port=port)