from dash import dcc

def dropdown():
    dropdown = dcc.Dropdown(
        id = "dropdown-quarter",
        options=[
            {'label': 'Q1', 'value': 'Q1'},
            {'label': 'Q2', 'value': 'Q2'},
            {'label': 'Q3', 'value': 'Q3'},
            {'label': 'Q4', 'value': 'Q4'},
        ],
        value="Q2",  # Default value
        clearable=False,
        style={'width': '50%', 'color': 'black'},
        placeholder="Select Quarter for Training",
    )
    return dropdown