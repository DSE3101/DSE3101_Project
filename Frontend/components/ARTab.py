from dash import html

def ARTab():
    ar_model_content = html.Div([
        html.H3("AR MODEL", style={'color' : 'lightblue'}),
        html.P("We will be using the 2012 Q2 prediction to explain our models"),
        html.P("An AR model is the simplest and most basic forecasting model in econometrics to predict future behaviour based on past data. This serves as a good benchmark for other models that includes different variations and methods in deciding the number of lags and variables in different ways. "),

        html.H4("How does the model work"),
        html.P("The first step to build the AR model was to choose the number of lags of GDP growth (Y), using leave-one-out-cross-validation (LOOCV). We iterate through 8 models, AR(1) to AR(8), and conduct LOOCV for every observation within each lag. We decided to stick with 8 as the highest number of lags as we do not expect persistence of more than 2 years if at all. We then proceed to find the mean squared error (MSE) for each lag by taking the average of the sum of squared errors. For all lags, we first have to ensure that the sample size of each dataset is the same. This can be dealt with by slicing the data from the bottom (which would be the more recent data) to follow the number of observations of the model with the least sample size and largest number of lags; in this case, that would be AR(8). After gathering the MSE for each lag, we would then choose the number of lags depending on the model with the lowest MSE."),
        html.P("The second step would then be to use the optimal lags to build the model to forecast. This consists of fitting the data with the optimal lags to a linear regression model."),
        html.P("Lastly, we do a 1-step forecast to predict Yt+1, forecasting one period ahead."),

        html.H4("Finding Yt+h"),
        html.P("To predict Yt+2, we carry out the exact same method but we iterate through AR(2) to AR(9) to carry out a 2-step forecast. We repeat the steps to get 8 forecasts from Yt+1 to Yt+8, sequentially building a different model to carry out h-step forecasts for each h in Yt+h."),

        html.H4("Why not AIC/BIC?"),
        html.P("Initially, we used the Akaike Information Criterion (AIC) to find the optimal number of lags for each model for each h-step forecast, which was a big mistake. AIC is unable to forecast predictions beyond Yt+1 as its likelihood function is only conditioned on the observed data up to the current point. Forecasting beyond 1-step requires forecasting on future data, which is not available at that time."),            
        
        html.H4("Calculating RMSFE"),
        html.P("To assess the performance of our models, we carried out backtesting for the models. Backtesting involves finding the root mean squared-error (RMSE), using LOOCV MSE. Since we have already done this step in creating AR and ADL models, we find the square root of the MSE for the optimal model for each h-step forecast.")
    ],style={'text-align': 'center'})

    return ar_model_content