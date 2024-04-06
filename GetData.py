import pickle
import pandas as pd

def get_data():
    # Load the macro variables
    with open('preprocessed_data.pkl', 'rb') as f:
        macro_variables = pickle.load(f)

    # Combine all variables for chosen quarter and remove rows
    def get_vintage_data(vintage_year, vintage_quarter, data_year, data_quarter):
        # Create YYQq to slice vintages by index
        vintages = ["65Q4"]
        vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
        vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
        vintages.extend([f'{i}Q{j}' for i in range(10, int(curr_year)) for j in range(1, 5)])
        vintages.extend([f'{int(curr_year)}Q{j}' for j in range(1, int(curr_quarter)+1)])

        year_quarter_index = vintages.index(f'{vintage_year[-2:]}Q{vintage_quarter}')
        df = []
        for var in macro_variables:
            df.append(var.iloc[:, year_quarter_index])
        df = pd.concat(df, axis=1)
        # Remove rows after chosen quarter
        df = df[df.index <= f"{data_year}:Q{data_quarter}"]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    # Get input
    # chosen_variable = macro_variables[macro_variables.index(input("Choose a macro variable to forecast"))]
    chosen_variable = macro_variables[-1]
    curr_year = chosen_variable.columns[-1][-4:-2]
    curr_quarter = chosen_variable.columns[-1][-1]
    # year_input = input("Choose real time data from 1966 to 2023")
    year_input = "2012"
    # quarter_input = input("Choose a quarter from 1 to 4")
    quarter_input = "2"
    # h_step_input = input("Choose number of steps to forecast")
    h_step_input = 12
    years_ahead = h_step_input // 4
    quarters_ahead = h_step_input % 8
    h_step_year = int(year_input) + years_ahead
    h_step_quarter = int(quarter_input) + quarters_ahead
    if h_step_quarter > 4:
        h_step_year += 1
        h_step_quarter -= 4
    h_step_year = str(h_step_year)
    h_step_quarter = str(h_step_quarter)

    # Slice data as needed
    real_time_X, real_time_y = get_vintage_data(year_input, quarter_input, year_input, quarter_input)
    real_time_X, real_time_y = real_time_X[:-1], real_time_y[:-1]
    latest_X, latest_y = get_vintage_data(curr_year, curr_quarter, h_step_year, h_step_quarter)
    latest_X_train = latest_X.iloc[:len(real_time_y), :]
    latest_y_train = latest_y.iloc[:len(real_time_y)]
    latest_X_test = latest_X.iloc[len(real_time_y):, :]
    latest_y_test = latest_y.iloc[len(real_time_y):]

    return real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter


