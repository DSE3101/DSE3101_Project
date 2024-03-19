import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
# from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

#num_lags = int(input("How many lags?\n"))
#year = input("Which year? (YY)\n")
#quarter = input("Which quarter? (1-4)\n")
num_lags = 30
year = "21"
quarter = "4"
period_t = "ROUTPUT" + year + "Q" + quarter

routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")

real_time_data = routput[period_t].dropna()
real_time_model = AutoReg(real_time_data, lags=num_lags)
real_time_model_fit = real_time_model.fit()
print(real_time_model_fit.summary())
lag_plot(real_time_data)
plt.show()
real_time_data.plot()
plt.show()
# autocorrelation_plot(real_time_data)
plot_acf(real_time_data, lags=num_lags)
plt.show()
real_time_result = adfuller(real_time_data)
print('ADF Statistic: %f' % real_time_result[0])
print('p-value: %f' % real_time_result[1])
print('Critical Values:')
for key, value in real_time_result[4].items():
 print('\t%s: %.3f' % (key, value))

row_slice = (2000 + int(year) - 1947) * 4 + 1 + int(quarter)
latest_vintage_data = routput.iloc[:row_slice, -1].dropna()
latest_vintage_model = AutoReg(latest_vintage_data, lags=num_lags)
latest_vintage_model_fit = latest_vintage_model.fit()
print(latest_vintage_model_fit.summary())
lag_plot(latest_vintage_data)
plt.show()
latest_vintage_data.plot()
plt.show()
# autocorrelation_plot(latest_vintage_data)
plot_acf(latest_vintage_data, lags=num_lags)
plt.show()
latest_vintage_result = adfuller(latest_vintage_data)
print('ADF Statistic: %f' % latest_vintage_result[0])
print('p-value: %f' % latest_vintage_result[1])
print('Critical Values:')
for key, value in latest_vintage_result[4].items():
 print('\t%s: %.3f' % (key, value))