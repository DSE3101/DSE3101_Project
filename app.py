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
from components.MLTab import *
#from components.EvalTab import EvalTab
from data import mainplot
import pickle


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
        #dcc.Tab(label='Evaluation', className="tab", children=EvalTab())
    ]),
    html.Div(id='tabs-content'),
    dcc.Store(id = 'year-quarter')
])

#Slider elements will update the graph
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

#Data taken from slider and quarter will be shown as a reactive statement in trainingtab
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

#Data taken from training tab will be called to AR, ADL and ML
@app.callback(
    Output('year-quarter', 'data'),
    [Input('date-slider', 'value'), Input('quarter-dropdown', 'value')]
)
def update_shared_data(date_slider_value, quarter_dropdown_value):
    # Logic to process the slider and dropdown values
    # For example, convert slider value to year and quarter to a specific format
    data = {
        'year': date_slider_value,
        'quarter': quarter_dropdown_value
    }
    return data

#Preprocess the data using the input and outputs

@app.callback(
    Output('X', 'y'),
    [Input('year-quarter', 'data')]
)
def trainML(year, quarter):
    with open('preprocessed_data.pkl', 'rb') as f:
        (RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
        rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
        DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
        NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
        P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
        RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT) = pickle.load(f)

    macro_variables = [RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
        rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
        DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
        NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
        P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
        RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT]
    # Create YYQq to slice vintages by index
    vintages = ["65Q4"]
    vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
    vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
    vintages.extend([f'{i}Q{j}' for i in range(10, 24) for j in range(1, 5)])
    vintages.extend(["24Q1"])

    #Get vintage
    quarter_index = vintages.index(f'{year[-2:]}Q{quarter}')
    df = []
    for var in macro_variables:
        df.append(var.iloc[:, quarter_index])
    df = pd.concat(df, axis=1)
    # Remove rows after chosen quarter
    df = df[df.index <= f"{year}:Q{quarter}"]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    real_time_X, real_time_y = get_vintage_data(year_input, quarter_input)
    latest_X, latest_y = get_vintage_data(ROUTPUT.columns[-1][-4:-2], ROUTPUT.columns[-1][-1])

    # Train Test Split
    latest_X_train, latest_X_test, latest_y_train, latest_y_test = train_test_split(latest_X, latest_y, test_size=0.2, random_state=42)

    # Train a random forest model
    latest_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    latest_rf_model.fit(latest_X_train, latest_y_train)

    # Get feature importance scores
    latest_feature_importance = latest_rf_model.feature_importances_

    # Create a DataFrame to store feature importance scores
    latest_feature_importance_df = pd.DataFrame({'Feature': latest_X.columns, 'Importance': latest_feature_importance})
    latest_feature_importance_df = latest_feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print feature importance scores
    print("Latest Feature Importance Scores:")
    print(latest_feature_importance_df)

    # Choose the top N variables based on feature importance
    top_n_variables = 21  # You can adjust this value based on your preference
    latest_selected_variables = latest_feature_importance_df.head(top_n_variables)['Feature'].tolist()
    print("\nTop", top_n_variables, "Latest Variables Selected:", latest_selected_variables)

    # Train a new random forest model using only the selected variables
    latest_X_train_selected = latest_X_train[latest_selected_variables]
    latest_X_test_selected = latest_X_test[latest_selected_variables]

    latest_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
    latest_rf_model_selected.fit(latest_X_train_selected, latest_y_train)

    # Evaluate the model's performance
    latest_y_pred = latest_rf_model_selected.predict(latest_X_test_selected)
    latest_mse = mean_squared_error(latest_y_test, latest_y_pred)

    # Train Test Split
    real_time_X_train, real_time_X_test, real_time_y_train, real_time_y_test = train_test_split(real_time_X, real_time_y, test_size=0.2, random_state=42)

    # Train a random forest model
    real_time_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    real_time_rf_model.fit(real_time_X_train, real_time_y_train)

    # Get feature importance scores
    real_time_feature_importance = real_time_rf_model.feature_importances_

    # Create a DataFrame to store feature importance scores
    real_time_feature_importance_df = pd.DataFrame({'Feature': real_time_X.columns, 'Importance': real_time_feature_importance})
    real_time_feature_importance_df = real_time_feature_importance_df.sort_values(by='Importance', ascending=False)


    # Choose the top N variables based on feature importance
    top_n_variables = 21  # You can adjust this value based on your preference
    real_time_selected_variables = real_time_feature_importance_df.head(top_n_variables)['Feature'].tolist()

    # Train a new random forest model using only the selected variables
    real_time_X_train_selected = real_time_X_train[real_time_selected_variables]
    real_time_X_test_selected = real_time_X_test[real_time_selected_variables]

    real_time_rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
    real_time_rf_model_selected.fit(real_time_X_train_selected, real_time_y_train)

    # Evaluate the model's performance
    real_time_y_pred = real_time_rf_model_selected.predict(real_time_X_test_selected)
    real_time_mse = mean_squared_error(real_time_y_test, real_time_y_pred)
    
    return latest_mse, real_time_feature_importance_df, top_n_variables, real_time_selected_variables, real_time_mse

if __name__ == '__main__':
    app.run_server(debug=True)
