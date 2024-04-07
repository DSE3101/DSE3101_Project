import numpy as np
import pandas as pd 
import pandas as pd
import plotly.graph_objects as go

def mainplot():
    routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
    real_time_data = routput["ROUTPUT24Q1"]
    routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)
    routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=routput['DATE'], y=routput['ROUTPUT24Q1'], mode='lines', name='ROUTPUT24Q1'))
    # Updating layout for readability
    fig.update_layout(title='ROUTPUT24Q1 Over Time',
                    xaxis_title='Time',
                    yaxis_title='ROUTPUT24Q1 Value',
                    hovermode="x")
    
    # Showing the figure
    return fig
    