import pandas as pd

# input_macro_var = input("Input macro variable to forecast:\n")
input_macro_var = ""
# input_year_real_time = input("Enter real time year (YYYY):\n")
input_year_real_time = "2003"
# input_month_real_time = input("Enter real time month (MM):\n")
input_month_real_time = "01"
# input_year_latest = input("Enter latest year (YYYY):\n")
input_year_latest = "2024"
# input_month_latest = input("Enter latest month (MM):\n")
input_month_latest = "03"
# h_step = input("Enter h-steps to forecast: \n")
h_step = 36


df_real_time = pd.read_csv(f'./data/project data/fred-md data/{input_year_real_time}-{input_month_real_time}.csv')
df_latest = pd.read_csv(f'./data/project data/fred-md data/{input_year_latest}-{input_month_latest}.csv')

transformation_row = df_latest.iloc[0, :]

df_real_time.drop(index=0, inplace=True)
df_latest.drop(index=0, inplace=True)

df_real_time["sasdate"] = pd.to_datetime(df_real_time["sasdate"])
df_latest["sasdate"] = pd.to_datetime(df_latest["sasdate"])

real_time_index = df_latest.index[df_latest["sasdate"] == f'{input_month_real_time}/1/{input_year_real_time}'][0]
df_real_time = df_real_time.iloc[:real_time_index, :]
df_latest_train = df_latest.iloc[:real_time_index, :]
df_latest_test = df_latest.iloc[real_time_index: real_time_index+h_step, :]



