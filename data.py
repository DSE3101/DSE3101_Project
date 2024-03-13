import numpy as np
import pandas as pd 
import pandas as pd

def get_date_range():
    return pd.date_range(start="1960-01-01", end="2023-12-31", freq='QE')

def generate_data():
    np.random.seed(123)
    date_range = pd.date_range(start="1960-01-01", end="2023-12-31", freq='QE')
    values = np.cumsum(np.random.uniform(-10, 10, len(date_range)))
    dummy_data = pd.DataFrame({'date': date_range, 'value': values})
    return dummy_data