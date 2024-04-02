import pandas as pd
import pickle

# Load the macro variables
with open('preprocessed_data.pkl', 'rb') as f:
    (RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS) = pickle.load(f)

macro_variables = [RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
     rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
     DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
     NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
     P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
     RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS]

# Create YYQq to slice vintages by index
vintages = ["65Q4"]
vintages.extend([f'{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
vintages.extend([f'0{i}Q{j}' for i in range(0, 10) for j in range(1, 5)])
vintages.extend([f'{i}Q{j}' for i in range(10, 24) for j in range(1, 5)])
vintages.extend(["24Q1"])

# year_input = input("Choose real time data from 1966 to 2023")
year_input = "1998"
# quarter_input = input("Choose a quarter from 1 to 4")
quarter_input = "3"
quarter_index = vintages.index(f'{year_input[-2:]}Q{quarter_input}')

# Combine all variables for chosen quarter
real_time_data = []
for var in macro_variables:
    real_time_data.append(var.iloc[:, quarter_index])
real_time_data = pd.concat(real_time_data, axis=1)

# Remove quarters after chosen quarter
print(real_time_data[real_time_data.index <= f"{year_input}:Q{quarter_input}"])