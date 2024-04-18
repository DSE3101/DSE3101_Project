from dash import dcc
import pandas as pd

date_range_yearly = pd.date_range(start='1966-01-01', end='2023-12-31', freq='YS')

def dropdown_year():
    date_range_yearly = pd.date_range(start='1966-01-01', end='2019-10-01', freq='YS')
    dropdown = dcc.Dropdown(
        id = "dropdown-year",
        options=[{'label': str(year.year), 'value': str(year.year)} for year in date_range_yearly],
        value = "1970", #Default value
        clearable=False,
        searchable=True,
        style={'width': '50%', 'color': 'black'},
        placeholder="Select Year for Training",
    )
    return dropdown
