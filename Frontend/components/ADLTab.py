from dash import html

def ADLTab():
    ADLTab = html.Div([
    html.H3("ADL MODEL", style={'color' : 'lightblue'}),
    html.P("We will be using the 2012 Q2 prediction to explain our models"),
    
    html.H4("How does the model work"),
    html.P("Our Autoregressive Distributed Lag (ADL) model extends on the concept of autoregression by using both the past values of GDP and other economic indicators to forecast GDP."),
    
    html.H4("Best ADL Function"),
    html.P("We chose 6 variables manually that we decided will have the most economical impact on the ADL model. More will be discussed in our technical documentation."),
    html.P("We run a nested iteration for all combinations of the 6 variables and compare the AIC. So we run every single permutation possible, from 1 variable to 6 variables. We will choose the smallest AIC value and return the respective variables."),
    html.Img(src="assets/adl_tab/variables.jpg", className="graph-image"),
    html.P("The model chosen the above-mentioned variables with 1 lag each. The extra 1 value is the lag of the Y value, which is growth rate."),
    
    html.H4("Expanding Window"),
    html.P("We will use the same model to run an expanding window of 1 step forecast each. Giving and to get a model to predict y+1, aka a one step ahead forecast. We then repeat this step 8 times, using the predicted y+1/y+2/… values all the way until y+7 to predict the y+8 forecast. This will generate 8 prediction values as shown."),
    html.Img(src="assets/adl_tab/prediction.jpg", className="graph-image"),
    
    html.H4("Calculating RMSFE"),
    html.P("We then calculated the RMSFE using the root mean squared error between the true y values and the predicted y values. In the case of 2012 Q2, the returned RMSFE value was.")
], style={'text-align': 'center', 'color': 'white'})

    return ADLTab