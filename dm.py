import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t

#Use DM to compare
def DM(model1_values, model2_values, test1_values, test2_values, h =8):
    e1 = test1_values - model1_values
    e2 = test2_values - model2_values
    d = np.abs(e1) - np.abs(e2)
    var_d = np.var(d)
    t_dm = np.mean(d)/np.squrt((1/len(d))*var_d)
    
    dof = len(d)-1
    data_size = len(test1_values)
    correction_factor = np.sqrt(1+ data_size**(-1) * (1-2*h) + data_size**(-2) * h*(h-1))
    t_hln = correction_factor * t_dm
    
    p_value = 2* t.cdf(-np.abs(t_dm), df = dof)
    
    return t_dm, t_hln, p_value
    
#interpreting this, if original dm stat is significant (p-value) -> significant diff in the predictive accuracy between 2 models
#If t_hln is significant, it strengthens the evidence for prediction 
#If p-value < 0.05: one model is significantly better/worse than the other
#If p-value is large: both models are kinda the same
