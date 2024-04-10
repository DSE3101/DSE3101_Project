import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def MLTab():
    MLTab = html.Div([
    html.H3("RF MODEL", style={'color' : 'lightblue'}),
    html.P("We will be using the 2012 Q2 prediction to explain our models"),
    
    html.H3("How it works"),
    html.P("We also used Random Forest (RF), a machine learning technique to determine which economic indicators have the most influence on GDP growth. All economic indicators are used to train the model, and optimal n indicators are selected based on their importance scores. The process is then repeated using the optimal n indicators to find which indicators influence GDP the most."),
    
    html.H3("Why n = 100?"),
    html.P("Too low n will lead to an underfitted model, while too high n will lead to a very noisy model, so we decided n=100 is a good fit for our project demands."),
    
    html.H3("Features Importance"),
    html.P("We then chose the top 6 most important features and used them for the RF model."),
    html.Img(src="assets/rf_tab/features.png", className="graph-image"),
    
    html.H3("Expanding Window"),
    html.P("Using the identified model, we will run one step forecasts using this model 8 times, using the predicted y+1/y+2/… values all the way until y+7 to predict the y+8 forecast. This will generate 8 prediction values as shown."),
    
    html.H3("Calculating RMSFE"),
    html.P("We then calculated the RMSFE using the root mean squared error between the true y values and the predicted y values. In the case of 2012 Q2, the returned RMSFE value was 0.006."),
], style={'text-align': 'center', 'color': 'white'})


    return MLTab
