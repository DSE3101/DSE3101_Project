import numpy as np
import pandas as pd 
import pandas as pd
import plotly.graph_objects as go

def mainplot():
    routput = pd.read_excel("data/project data/ROUTPUTQvQd.xlsx", na_values="#N/A")
    real_time_data = routput["ROUTPUT21Q4"]

    # Replace the colon in the 'DATE' column strings
    routput['DATE'] = routput['DATE'].str.replace(':', '', regex=True)

    # Create a PeriodIndex from the 'DATE' column with quarterly frequency
    routput['DATE'] = pd.PeriodIndex(routput['DATE'], freq='Q').to_timestamp()
    
    # Creating a Plotly figure
    fig = go.Figure()

    # Adding a trace for the ROUTPUT21Q4 data
    fig.add_trace(go.Scatter(x=routput['DATE'], y=routput['ROUTPUT21Q4'], mode='lines', name='ROUTPUT21Q4'))

    # Updating layout for readability
    fig.update_layout(title='ROUTPUT21Q4 Over Time',
                    xaxis_title='Time',
                    yaxis_title='ROUTPUT21Q4 Value',
                    hovermode="x")

    # Showing the figure
    return fig
    