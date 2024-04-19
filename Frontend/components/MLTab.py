from dash import html


def MLTab():
    MLTab = html.Div([
    html.H3("RF MODEL", style={'color' : 'lightblue'}),
    html.P("We will be using the 2012 Q2 prediction to explain our models"),
    
    html.H3("How it works"),
    html.P("In using RF, each tree represents different combinations of macroeconomic variables, representing information from different parts of the economy’s current state. Each tree is random, selecting a subset of different macroeconomic variables to be trained on. Each tree would have a feature importance in forecasting GDP growth, with a higher value corresponding to a higher importance in contributing to GDP growth."),
    html.P("To get a one-step forecast, we first train the RF model with all the variables, and allow the RF algorithm to iterate through every possible decision tree to find all the various feature importance. The decision trees would comprise different combinations of 1 lag of each macroeconomic variable. We then select the top 6 most important variables with the highest feature importance score, and train and fit the new RF model with these selected variables. Lastly, we would then predict one step ahead to get the forecast for Yt+1."),
    html.P("To get the two-step forecast, we would train a new RF model using 2 lags of each macroeconomic variable, repeating the same steps as the one-step forecast. The RF model may choose 6 new choices of important variables in which we would train and fit a new RF model. This step is repeated for every h-step forecast from 1-step to 8-steps."),

    html.H3("Why n = 100?"),
    html.P("Too low n will lead to an underfitted model, while too high n will lead to a very noisy model, so we decided n=100 is a good fit for our project demands."),
    
    html.H3("Why random state = 42"),
    html.P("There is no specific reason to set random_state = 42 over any other parameter, the purpose of setting seed is to ensure reproducibility, to get the same sequence of numbers each time we run this code. This is helpful with debugging, sharing results, and comparing different models."),
        
    html.H3("Calculating RMSFE"),
    html.P("For the RF model, since we did not manually carry out variable selection through LOOCV, we have to then carry out the procedure of: Doing LOOCV, Finding MSE for the optimal model for every h-step forecast, Finding RMSE for every h-step forecast"),
], style={'text-align': 'center', 'color': 'white'})


    return MLTab
