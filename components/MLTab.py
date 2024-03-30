import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def phil_month_to_quarter_cigxm(df):
    df["DATE"] = df["DATE"].str.replace(":", "-")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

# cpi, h_starts, m1, pcpi, unemployment_rate
def phil_month_to_quarter_others(df):
    df["DATE"] = df["DATE"].str.replace(":", "-")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.month % 3 == 0].copy()
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

def fred_month_to_quarter(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"].dt.month % 3 == 0].copy()
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

def fred_month_to_quarter_ir_3m(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["quarter"] = df["DATE"].dt.to_period("Q")
    return df

def load_and_preprocess_data():
    csr_conf = pd.read_csv("./data/project data/Consumer Confidence Indicator.csv")
    csr_conf = fred_month_to_quarter(csr_conf)
    cpi = pd.read_csv("./data/project data/CPI.csv")
    cpi = phil_month_to_quarter_others(cpi)
    h_starts = pd.read_csv("./data/project data/Housing Starts.csv")
    h_starts = phil_month_to_quarter_others(h_starts)
    ir_3m = pd.read_csv("./data/project data/Interest Rates 3M.csv")
    ir_3m = fred_month_to_quarter_ir_3m(ir_3m)
    ir_10y = pd.read_csv("./data/project data/Interest Rates 10Y.csv")
    ir_10y = fred_month_to_quarter(ir_10y)
    m1 = pd.read_csv("./data/project data/M1.csv")
    m1 = phil_month_to_quarter_others(m1)
    pcpi = pd.read_csv("./data/project data/PCPI.csv")
    pcpi = phil_month_to_quarter_others(pcpi)
    consumption = pd.read_csv("./data/project data/Real Consumption.csv")
    consumption = phil_month_to_quarter_cigxm(consumption)
    exports = pd.read_csv("./data/project data/Real Export.csv")
    exports = phil_month_to_quarter_cigxm(exports)
    govt_spending = pd.read_csv("./data/project data/Real Govt Spending.csv")
    govt_spending = phil_month_to_quarter_cigxm(govt_spending)
    imports = pd.read_csv("./data/project data/Real Import.csv")
    imports = phil_month_to_quarter_cigxm(imports)
    investments = pd.read_csv("./data/project data/Real Non-Residential Investment.csv")
    investments = phil_month_to_quarter_cigxm(investments)
    gdp = pd.read_csv("./data/project data/Real Output.csv")
    gdp = phil_month_to_quarter_cigxm(gdp)
    unemployment_rate = pd.read_csv("./data/project data/Unemployment Rate.csv")
    unemployment_rate = phil_month_to_quarter_others(unemployment_rate)
    
    # test with real time first
    # test with 60 most recent quarters
    # err some 2023Q4 some 2024Q1
    # just testing random forest first, remember to fix the dates
    csr_conf_latest = csr_conf.iloc[-60:,-2].reset_index(drop=True)
    cpi_latest = cpi.iloc[-60:,-2].reset_index(drop=True)
    h_starts_latest = h_starts.iloc[-60:, -2].reset_index(drop=True)
    ir_3m_latest = ir_3m.iloc[-60:, -2].reset_index(drop=True)
    ir_10y_latest = ir_10y.iloc[-60:, -2].reset_index(drop=True)
    m1_latest = m1.iloc[-60:, -2].reset_index(drop=True)
    pcpi_latest = pcpi.iloc[-60:, -2].reset_index(drop=True)
    consumption_latest = consumption.iloc[-60:, -2].reset_index(drop=True)
    exports_latest = exports.iloc[-60:, -2].reset_index(drop=True)
    govt_spending_latest = govt_spending.iloc[-60:, -2].reset_index(drop=True)
    imports_latest = imports.iloc[-60:, -2].reset_index(drop=True)
    investments_latest = investments.iloc[-60:, -2].reset_index(drop=True)
    gdp_latest = gdp.iloc[-60:, -2].reset_index(drop=True)
    unemployment_rate_latest = unemployment_rate.iloc[-60:, -2].reset_index(drop=True)


    macro_variables = [csr_conf_latest, cpi_latest, h_starts_latest, ir_3m_latest, ir_10y_latest, m1_latest, pcpi_latest, consumption_latest, exports_latest, govt_spending_latest, imports_latest, investments_latest, unemployment_rate_latest]
    X = pd.concat(macro_variables, axis=1, ignore_index=True)
    X = X.replace(',', '', regex=True)
    X = X.replace(' ', '', regex=True)
    X_headers = ["Consumer Confidence", "Consumer Price Index (CPI)", 
                 "Housing Starts", "Interest Rates (3 months)", 
                 "Interest Rates (10 years)", "Money 1", 
                 "Producer Consumer Price Index", "Consumption", 
                 "Exports", "Government Spendings", 
                 "Imports", "Investments", 
                 "Unemployment Rates"]
    X.columns = X_headers
    X = X.astype(float)
    y = gdp_latest
    y = y.replace(',', '', regex=True)
    y = y.replace(' ', '', regex=True)
    y.columns = ["gdp"]
    y = y.astype(float)
    
    return X, y

def train_and_evaluate_ml_model(X, y):
     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importance scores
    feature_importance = rf_model.feature_importances_

    # Create a DataFrame to store feature importance scores
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Choose the top N variables based on feature importance
    top_n_variables = 5  # You can adjust this value based on your preference
    selected_variables = feature_importance_df.head(top_n_variables)['Feature'].tolist()

    # Train a new random forest model using only the selected variables
    X_train_selected = X_train[selected_variables]
    X_test_selected = X_test[selected_variables]

    rf_model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_selected.fit(X_train_selected, y_train)

    # Evaluate the model's performance
    y_pred = rf_model_selected.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    return feature_importance_df, mse


def MLTab():
    X, y = load_and_preprocess_data()

    # Train the model and get feature importance and MSE
    feature_importance_df, mse = train_and_evaluate_ml_model(X, y)

    # Sort the DataFrame by importance and take the top 5
    top_5_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(5)

    # Manually construct the table to allow for custom styling
    table_header = [
        html.Thead(html.Tr([html.Th("Feature"), html.Th("Importance")]))
    ]
    table_body = [html.Tbody([
        html.Tr([
            html.Td(feature),
            html.Td(f"{importance:.4f}", style={"font-weight": "bold" if index < 5 else ""})
        ]) for index, (feature, importance) in enumerate(zip(feature_importance_df['Feature'], feature_importance_df['Importance']))
    ])]

    # Prepare reactive text for the top 5 features
    top_features_text = "Top 5 Features: " + ", ".join(top_5_features['Feature'].tolist())

    # Prepare content for the ML tab
    MLTab = html.Div([
        html.H3("Machine Learning Analysis"),
        html.P("This section showcases the results of the machine learning analysis, including feature importance and model performance."),
        html.H4("Feature Importance"),
        dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True),
        html.H4(f"Mean Squared Error (MSE): {mse:.2f}"),
        html.H5(top_features_text)
    ])

    return MLTab