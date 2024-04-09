import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t

#Use for all 3
def RMSE(model_values, test_values):
    rmse_value = 4
    return rmse_value



def EvalTab():
    #rmse_ar = RMSE(ar_values, test_values)
    #rmse_adl = RMSE(adl_values, test_values)
    #rmse_forest = RMSE(forest_values, test_values)
    #dm_test = DM(adl_values, forest_values, test_values, h = 8)
    EvalTab =  html.Div([
        html.Div(id='evaluation-results'),
        html.H3("Evaluating our models"),
        html.P("We will use 2 tests to determine which model is the most appropriate"),
        html.P("The tests are RMSE and Diebold-Mariano Test"),
        html.H4("RMSE"),
        html.P("RMSE is an extremely simple and easy to implement test"),
        ##Insert 3 graphs, which are 3 fancharts, and with the test data
        html.H5("AR Model"),
        #Graph 1
        html.P("The AR model serves as our baseline model. Based on econometric theory, it shoould be the worst performing in terms of RMSE values"),
        #html.P("Running the AR model described above, the RMSE value is" + rmse_ar),
        html.P("The value is quite high, so we will see how our next model fare"),
        html.P("Based on our models, the model with the lowest RMSE is xxx"),
        html.H5("ADL Model"),
        #Graph 2
        #html.P("The ADL model will return us a RMSE value of " + rmse_adl), #Text will be on the right
        html.H5("Regression Forest Model"),
        #Graph 3
        #html.P("Last but not least, running our own regression model returns us the RMSE value of " + rmse_forest),
        
        html.P("But using the RMSE has it's own issues,"),
        html.P("For instance, RMSE are extremely sensitive to outliers, and is not as statistically sound as our other test."),
        html.P("This is where our next test comes in"),

        html.H4("Diebold-Mariano (DM) Test"),
        html.P("We will now run the DM test between the ADL and regression forest model."), 
        html.P("The reason why we are not running this on the AR model is because we have already established that the AR model is simply a benchmark model.")
    ])