import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t

#Use DM to compare
def DM(model1_values, model2_values, test_values, h =12):
    
    test_values = (test_values.iloc[:,0])
    # Calculate forecast errors
    e1 = test_values - model1_values
    e2 = test_values - model2_values
    eps = 0.0000000001


    d = np.abs(e1) - np.abs(e2)
    var_d = np.var(d)
    t_dm = np.mean(d)/np.sqrt((1/len(d))*(var_d+eps))
    
    dof = len(d)-1
    data_size = len(test_values)
    correction_factor = np.sqrt(1+ data_size**(-1) * (1-2*h) + data_size**(-2) * h*(h-1))
    t_hln = correction_factor * t_dm
    
    p_value = 2* t.cdf(-np.abs(t_dm), df = dof)
    
    return t_dm, t_hln, p_value
    
#interpreting this, if original dm stat is significant (p-value) -> significant diff in the predictive accuracy between 2 models
#If t_hln is significant, it strengthens the evidence for prediction 
#If p-value < 0.05: one model is significantly better/worse than the other
#If p-value is large: both models are kinda the same
