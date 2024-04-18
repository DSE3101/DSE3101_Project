import pandas as pd
import numpy as np
from scipy.stats import t

#Use DM to compare
def DM(model1_values, model2_values, test_values, h =8, p =8):
    
    test_values = test_values.iloc[:,0]
    # Calculate forecast errors
    e1 = test_values - model1_values
    e2 = test_values - model2_values
    eps = 0.0000000001


    d = np.abs(e1) - np.abs(e2)
    var_d = np.var(d)
    t_dm = np.mean(d)/np.sqrt((1/len(d))*(var_d+eps))
    
    dof = len(d)-1
    p_value = 2* t.cdf(-np.abs(t_dm), df = dof)
    
    return t_dm, p_value
    
#interpreting this, if original dm stat is significant (p-value) -> significant diff in the predictive accuracy between 2 models
#If p-value < 0.05: one model is significantly better/worse than the other
#If p-value is large: both models are kinda the same
