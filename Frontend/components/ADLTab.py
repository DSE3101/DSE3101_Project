from dash import html

def ADLTab():
    ADLTab = html.Div([
    html.H3("ADL MODEL", style={'color' : 'lightblue'}),
    html.P("We will be using the 2012 Q2 prediction to explain our models"),
    
    html.H4("How does the model work"),
    html.P("The ADL model is another simple model used in econometrics. Generally, following Occam’s razor is the most favourable method by keeping a model as simple as possible. We were inspired by the London School of Economics’s top-down approach, starting with several regressors then factoring it down to include only the important variables. We decided that the model was useful in capturing the dynamic effects of past values of our independent variable on the dependent variables, especially when we had a good understanding of the dependent variables and wanted to model them. "),
    
    html.H4("Variables seclected"),
    html.P("We chose 6 variables manually that we decided will have the most economical impact on the ADL model. More will be discussed in our technical documentation."),
    html.P("After examining all 65 dependent variables, we decided to include only 6, ensuring extensive and in-depth coverage from all the different categories such as monetary and financial, price level indices and the labour market. The 6 variables were: Consumer Price Index (CPI) under , Unemployment Rate (RUC) under , M1 Money Stock (M1), Housing Starts (HSTARTS), Industrial Production Index: Total (IPT). Some vintages were missing a number of observations. Namely for some of the chosen variables, when observations were missing, the available observations seemed to indicate no change on values from the next year. Thus, we assumed the the value would be the same and filled the NAs where necessary based on the next earliest datapoint for that observation. The implementation for the ADL model is relatively similar to the AR model, but it includes additional variables and their lags. We also knew that adding in more regressors (lag values) would lose one degree of freedom for each one, making statistical inference shaky. Having fewer regressors is one of the strengths of the ADL model, so we decided to leverage on that."),
    
    html.H4("Building the model"),
    html.P("Building the ADL model consists of: Finding the optimal number of lags using MSE, LOOCV for each lag of each variable; Fitting the lagged data with the optimal number of lags for each variable; Building the model for Yt+h|t; Forecasting Yt+h|t. Initially, we wanted to include all the 65 variables in the model before we choose which variables would potentially be important based on performance metrics such as MSE. However, this was unsuccessful as the time complexity to run the code chunk was extremely large, making running the ADL model slow. "),
    
    html.H4("Future improvements"),
    html.P("A failure in our ADL model was that we assumed all variables would be chosen, and we only needed to find the optimal number of lags for each variable. This was because going through all possible combinations of variables was too computationally expensive. Another failure with our model was that it determines the number of lags of each variable sequentially. This means that the optimal number of lags of one dependent variable (e.g. CPI) is chosen assuming no lags of other variables, followed by choosing the optimation number of lags of the next dependent variable (e.g. RUC) is chosen assuming no lags of other variables, which affects the optimal lags of CPI . Checking for this was also too computationally expensive."),
    
    html.H4("Calculating RMSFE"),
    html.P("To assess the performance of our models, we carried out backtesting for the models. Backtesting involves finding the root mean squared-error (RMSE), using LOOCV MSE. Since we have already done this step in creating AR and ADL models, we find the square root of the MSE for the optimal model for each h-step forecast.")
], style={'text-align': 'center', 'color': 'white'})

    return ADLTab