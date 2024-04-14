import pickle
import pandas as pd

def get_data(year_input, quarter_input):
    # Load the macro variables
    with open('preprocessed_data.pkl', 'rb') as f:
        macro_variables = pickle.load(f)

    # Combine all variables for chosen quarter and remove rows
    def get_vintage_data(vintage_year, vintage_quarter, data_year, data_quarter, chosen_var):
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
        # df = df[df.index <= f"{data_year}:Q{data_quarter}"]
        df = df[df.index < f"{data_year}:Q{data_quarter}"]
        y = df.loc[:, [f'{chosen_var}{vintage_year[-2:]}Q{vintage_quarter}']]
        X = df.drop(f'{chosen_var}{vintage_year[-2:]}Q{vintage_quarter}', axis=1)
        return X, y

    # Get input
    macro_variable_names = ["RCON", "rcong", "RCONND", "RCOND", "RCONS", "rconshh", "rconsnp", "rinvbf", "rinvresid",
                            "rinvchi", "RNX", "REX", "RIMP", "RG", "RGF", "RGSL", "rconhh", "WSD", "OLI", "PROPI", "RENTI",
                            "DIV", "PINTI", "TRANR", "SSCONTRIB", "NPI", "PTAX", "NDPI", "NCON", "PINTPAID", "TRANPF",
                            "NPSAV", "RATESAV", "NCPROFAT", "NCPROFATW", "M1", "M2", "CPI", "PCPIX", "PPPI", "PPPIX",
                            "P", "PCON", "pcong", "pconshh", "pconsnp", "pconhh", "PCONX", "PIMP", "POP", "LFC", "LFPART",
                            "RUC", "EMPLOY", "H", "HG", "HS", "OPH", "ULC", "IPT", "IPM", "CUT", "CUM", "HSTARTS", "ROUTPUT"]
    
    # chosen_variable_name = input("Choose a macro variable to forecast:\n")
    chosen_variable_name = "ROUTPUT"
    # year_input = input("Choose real time data from 1966 to 2023:\n")
    # year_input = "2012"
    # quarter_input = input("Choose a quarter from 1 to 4:\n")
    # quarter_input = "2"
    # h_step_input = int(input("Choose number of steps to forecast:\n")) - 1
    h_step_input = 8
    # curr_year = input("Choose latest time period (YYYY):\n")
    curr_year = "2020"
    # curr_quarter = input("Choose latest quarter (QQ):\n")
    curr_quarter = "1"


    years_ahead = h_step_input // 4
    quarters_ahead = h_step_input % 4
    h_step_year = int(year_input) + years_ahead
    h_step_quarter = int(quarter_input) + quarters_ahead
    if h_step_quarter > 4:
        h_step_year += 1
        h_step_quarter -= 4
    h_step_year = str(h_step_year)
    h_step_quarter = str(h_step_quarter)

    # Slice data as needed
    real_time_X, real_time_y = get_vintage_data(year_input, quarter_input, year_input, quarter_input, chosen_variable_name)
    latest_X, latest_y = get_vintage_data(curr_year, curr_quarter, h_step_year, h_step_quarter, chosen_variable_name)
    # In ROUTPUT, the vintages in 1992 did not revise values from 1947Q1 to 1958Q4
    # if (year_input == "1992") or (year_input == "1999" and quarter_input == "4") or (year_input == "2000" and quarter_input == "1"):
    #     slice_before_1959_Q1 = real_time_X.index.get_loc("1959:Q1")
    #     latest_X_train = latest_X.iloc[slice_before_1959_Q1:len(real_time_y), :]
    #     latest_y_train = latest_y.iloc[slice_before_1959_Q1:len(real_time_y)]
    #     latest_X_test = latest_X.iloc[len(real_time_y):, :]
    #     latest_y_test = latest_y.iloc[len(real_time_y):]
    #     real_time_X = real_time_X.iloc[slice_before_1959_Q1:len(real_time_y), :]
    #     real_time_y = real_time_y.iloc[slice_before_1959_Q1:len(real_time_y), :]
    # elif (year_input == "1996") or (year_input == "1997" and quarter_input == "1"):
    #     slice_before_1959_Q3 = real_time_X.index.get_loc("1959:Q3")
    #     latest_X_train = latest_X.iloc[slice_before_1959_Q3:len(real_time_y), :]
    #     latest_y_train = latest_y.iloc[slice_before_1959_Q3:len(real_time_y)]
    #     latest_X_test = latest_X.iloc[len(real_time_y):, :]
    #     latest_y_test = latest_y.iloc[len(real_time_y):]
    #     real_time_X = real_time_X.iloc[slice_before_1959_Q3:len(real_time_y), :]
    #     real_time_y = real_time_y.iloc[slice_before_1959_Q3:len(real_time_y), :]
    # else:
    #     latest_X_train = latest_X.iloc[:len(real_time_y), :]
    #     latest_y_train = latest_y.iloc[:len(real_time_y)]
    #     latest_X_test = latest_X.iloc[len(real_time_y):, :]
    #     latest_y_test = latest_y.iloc[len(real_time_y):]
    latest_X_train = latest_X.iloc[len(real_time_X)-48:len(real_time_X), :]
    latest_y_train = latest_y.iloc[len(real_time_y)-48:len(real_time_y)]
    latest_X_test = latest_X.iloc[len(real_time_X):, :]
    latest_y_test = latest_y.iloc[len(real_time_y):]
    real_time_X = real_time_X.iloc[len(real_time_X)-48:, :]
    real_time_y = real_time_y.iloc[len(real_time_y)-48:]
    # print(real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test)

    return real_time_X, real_time_y, latest_X_train, latest_y_train, latest_X_test, latest_y_test, curr_year, curr_quarter