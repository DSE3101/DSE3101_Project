# Data downloaded as of 31 Mar 2024
# Used quarterly real data
# Excluded discontinued variables
# Monthly data is converted into quarterly by taking the value of the middle month
# #N/A data replaced with -1

import pandas as pd
import numpy as np
import pickle

def pad_columns_to_09Q3(df, name):
    na_cols = [f'{name}65Q4']
    na_cols.extend([f'{name}{i}Q{j}' for i in range(66, 100) for j in range(1, 5)])
    na_cols.extend([f'{name}0{i}Q{j}' for i in range(0, 9) for j in range(1, 5)])
    na_cols.extend([f'{name}09Q1', f'{name}09Q2'])
    na_df = pd.DataFrame(columns=na_cols)
    df = pd.concat([na_df, df], axis=1)
    df.index.names = ['DATE']
    df.fillna(-999, inplace=True)
    return df

def pad_columns_to_98Q4(df, name):
    na_cols = [f'{name}65Q4']
    na_cols.extend([f'{name}{i}Q{j}'for i in range(66, 98) for j in range(1, 5)])
    na_cols.extend([f'{name}98Q1', f'{name}98Q2', f'{name}98Q3'])
    na_df = pd.DataFrame(columns=na_cols)
    df = pd.concat([na_df, df], axis=1)
    df.index.names = ["DATE"]
    df.fillna(-999, inplace=True)
    return df

def month_to_quarter(df, name):
    # Filter middle months
    col_months = df.columns
    col_middle_months = []
    for month in col_months:
        if month[-2:] in ("M2", "M5", "M8", "11"):
            col_middle_months.append(month)
    df = df[col_middle_months]

    # Convert month to quarter
    col_quarters = []
    for col in col_middle_months:
        month = col[-2:]
        if month == "M2":
            col_quarters.append(col[:-2] + "Q1")
        elif month == "M5":
            col_quarters.append(col[:-2] + "Q2")
        elif month == "M8":
            col_quarters.append(col[:-2] + "Q3")
        elif month == "11":
            col_quarters.append(col[:-3] + "Q4")
    df.columns = col_quarters

    return df

def rows_to_quarter(df):
    middle_months = []
    for year in range(1947, 2024):
        for month in (2, 5, 8):
            middle_months.append(f'{year}:0{month}')
        middle_months.append(f'{year}:11')
    df = df.loc[middle_months]

    time_quarters = [f'{year}:Q{quarter}' for year in range(1947, 2024) for quarter in range(1, 5)]
    df.index = time_quarters
    return df

# region NIPA Product Side - Real

# Real Personal Consumption Expenditures: Total (RCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcon
RCON = pd.read_excel("./data/project data/RCONQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Personal Consumption Expenditures: Goods (RCONG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcong
rcong = pd.read_excel("./data/project data/rcongQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
rcong = pad_columns_to_09Q3(rcong, "rcong")

# Real Personal Consumption Expenditures: Nondurable Goods (RCONND)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconnd
RCONND = pd.read_excel("./data/project data/RCONNDQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Personal Consumption Expenditures: Durable Goods (RCOND)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcond
RCOND = pd.read_excel("./data/project data/RCONDQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Personal Consumption Expenditures: Services (RCONS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcons
RCONS = pd.read_excel("./data/project data/RCONSQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Household Consumption Expenditures for Services (RCONSHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconshh
rconshh = pd.read_excel("./data/project data/rconshhQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
rconshh = pad_columns_to_09Q3(rconshh, "rconshh")

# Real Final Consumption Expenditures of NPISH (RCONSNP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconsnp
rconsnp = pd.read_excel("./data/project data/rconsnpQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
rconsnp = pad_columns_to_09Q3(rconsnp, "rconsnp")

# Real Gross Private Domestic Investment: Nonresidential (RINVBF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvbf
rinvbf = pd.read_excel("./data/project data/rinvbfQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)

# Real Gross Private Domestic Investment: Residential (RINVRESID)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvresid
rinvresid = pd.read_excel("./data/project data/rinvresidQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)

# Real Gross Private Domestic Investment: Change in Private Inventories (RINVCHI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvchi
rinvchi = pd.read_excel("./data/project data/rinvchiQvQd.xlsx", index_col="DATE").fillna(-999)

# Real Net Exports of Goods and Services (RNX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rnx
RNX = pd.read_excel("./data/project data/RNXQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Real Exports of Goods and Services (REX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rex
REX = pd.read_excel("./data/project data/REXQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Imports of Goods and Services (RIMP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rimp
RIMP = pd.read_excel("./data/project data/RIMPQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Government Consumption & Gross Investment: Total (RG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rg
RG = pd.read_excel("./data/project data/RGQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# Real Government Consumption & Gross Investment: Federal (RGF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rgf
RGF = pd.read_excel("./data/project data/RGFQvQd.xlsx", index_col="DATE").fillna(-999)

# Real Government Consumption & Gross Investment: State and Local (RGSL)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rgsl
RGSL = pd.read_excel("./data/project data/RGSLQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)

# endregion

# region NIPA By Major Function - Real

# Real Household Consumption Expenditures (RCONHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconhh
rconhh = pd.read_excel("./data/project data/rconhhQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
rconhh = pad_columns_to_09Q3(rconhh, "rconhh")

# Real Final Consumption Expenditures of NPISH (RCONSNP)
# Same as above

# endregion

# region NIPA Personal Income Side

# Wage and Salary Disbursements (WSD)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/wsd
WSD = pd.read_excel("./data/project data/wsdQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Other Labor Income (OLI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/oli
OLI = pd.read_excel("./data/project data/oliQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Proprietors' Income (PROPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/propi
PROPI = pd.read_excel("./data/project data/propiQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Rental Income of Persons (RENTI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/renti
RENTI = pd.read_excel("./data/project data/rentiQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Dividends (DIV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/div
DIV = pd.read_excel("./data/project data/divQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Personal Interest Income (PINTI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pinti
PINTI = pd.read_excel("./data/project data/pintiQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Transfer Payments (TRANR)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/tranr
TRANR = pd.read_excel("./data/project data/tranrQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Personal Contributions for Social Insurance (SSCONTRIB)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/sscontrib
SSCONTRIB = pd.read_excel("./data/project data/sscontribQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Nominal Personal Income (NPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/npi
NPI = pd.read_excel("./data/project data/npiQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Personal Tax & Nontax Payments (PTAX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ptax
PTAX = pd.read_excel("./data/project data/ptaxQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Nominal Disposable Personal Income (NDPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ndpi
NDPI = pd.read_excel("./data/project data/ndpiQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Nominal Personal Consumption Expenditures (NCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncon
NCON = pd.read_excel("./data/project data/nconQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Interest Paid by Consumers (PINTPAID)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pintpaid
PINTPAID = pd.read_excel("./data/project data/pintpaidQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Personal Transfer Payments to Foreigners (TRANPF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/tranpf
TRANPF = pd.read_excel("./data/project data/tranpfQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Nominal Personal Saving (NPSAV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/npsav
NPSAV = pd.read_excel("./data/project data/npsavQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Personal Saving Rate, Constructed (RATESAV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ratesav
RATESAV = pd.read_excel("./data/project data/ratesavQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# endregion

# region NIPA Other Income

# Nominal Corporate Profits After Tax Without IVA/CCAdj (NCPROFAT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncprofat
NCPROFAT = pd.read_excel("./data/project data/NCPROFATQvQd.xlsx", index_col="DATE").diff().fillna(-999)

# Nominal Corporate Profits After Tax With IVA/CCAdj (NCPROFATW)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncprofatw
NCPROFATW = pd.read_excel("./data/project data/NCPROFATWQvQd.xlsx", index_col="DATE").diff()
NCPROFATW.drop(index=["1946:Q1", "1946:Q2", "1946:Q3", "1946:Q4"], inplace=True)
na_cols_NCPROFATW = [f'NCPROFATW65Q4']
na_cols_NCPROFATW.extend([f'NCPROFATW{i}Q{j}' for i in range(66, 81) for j in range (1, 5)])
na_df_NCPROFATW = pd.DataFrame(columns=na_cols_NCPROFATW)
NCPROFATW = pd.concat([na_df_NCPROFATW, NCPROFATW], axis=1)
NCPROFATW.index.names = ["DATE"]
NCPROFATW.fillna(-999, inplace=True)

# endregion

# region Monetary and Financial

# M1 Money Stock (M1)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/m1
M1 = pd.read_excel("./data/project data/m1QvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)
M1 = rows_to_quarter(M1)

# M2 Money Stock (M2)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/m2
M2 = pd.read_excel("./data/project data/m2QvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().fillna(-999)
M2 = rows_to_quarter(M2)

# endregion

# region Price Level Indices

# Consumer Price Index, Quarterly Vintages (CPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cpi
CPI = pd.read_excel("./data/project data/cpiQvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)
CPI = rows_to_quarter(CPI)

# Core Consumer Price Index (PCPIX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpix
PCPIX = pd.read_excel("./data/project data/pcpixMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
PCPIX = month_to_quarter(PCPIX, "PCPIX")
PCPIX = pad_columns_to_98Q4(PCPIX, "PCPIX")
PCPIX = rows_to_quarter(PCPIX)

# Producer Price Index, Final Demand Finished Goods (PPPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pppi
PPPI = pd.read_excel("./data/project data/pppiMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
PPPI = month_to_quarter(PPPI, "PPPI")
PPPI = pad_columns_to_98Q4(PPPI, "PPPI")
PPPI = rows_to_quarter(PPPI)

# Core Producer Price Index, Final Demand Finished Goods (PPPIX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pppix
PPPIX = pd.read_excel("./data/project data/pppixMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
PPPIX = month_to_quarter(PPPIX, "PPPIX")
PPPIX = pad_columns_to_98Q4(PPPIX, "PPPIX")
PPPIX = rows_to_quarter(PPPIX)

# Price Index for GNP/GDP (P)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/p
P = pd.read_excel("./data/project data/PQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)

# Price Index for Personal Consumption Expenditures, Constructed (PCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcon
PCON = pd.read_excel("./data/project data/pconQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)

# Price Index for Personal Consumption Expenditures: Goods (PCONG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcong
pcong = pd.read_excel("./data/project data/pcongQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
pcong = pad_columns_to_09Q3(pcong, "pcong")

# Price Index for Household Consumption Expenditures for Services (PCONSHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconshh
pconshh = pd.read_excel("./data/project data/pconshhQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
pconshh = pad_columns_to_09Q3(pconshh, "pconshh")

# Price Index for Final Consumption Expenditures of NPISH (PCONSNP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconsnp
pconsnp = pd.read_excel("./data/project data/pconsnpQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
pconsnp = pad_columns_to_09Q3(pconsnp, "pconsnp")

# Price Index for Household Consumption Expenditures (PCONHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconhh
pconhh = pd.read_excel("./data/project data/pconhhQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
pconhh = pad_columns_to_09Q3(pconhh, "pconhh")

# Core Price Index for Personal Consumption Expenditures (PCONX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconx
PCONX = pd.read_excel("./data/project data/PCONXQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff()
na_cols_PCONX = ["PCONX65Q4"]
na_cols_PCONX.extend([f'PCONX{i}Q{j}'for i in range(66, 96) for j in range(1, 5)])
na_df_PCONX = pd.DataFrame(columns=na_cols_PCONX)
PCONX = pd.concat([na_df_PCONX, PCONX], axis=1)
PCONX.index.names = ["DATE"]
PCONX.fillna(-999, inplace=True)

# Price Index for Imports of Goods and Services (PIMP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pimp
PIMP = pd.read_excel("./data/project data/pimpQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff().diff().fillna(-999)

# endregion

# region Labour Market

# Civilian Noninstitutional Population, 16+ (POP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pop
POP = pd.read_excel("./data/project data/popMvMd.xlsx", index_col="DATE").diff()
POP = month_to_quarter(POP, "POP")
POP = pad_columns_to_98Q4(POP, "POP")
POP = rows_to_quarter(POP)

# Civilian Labor Force, 16+ (LFC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/lfc
LFC = pd.read_excel("./data/project data/lfcMvMd.xlsx", index_col="DATE").diff()
LFC = month_to_quarter(LFC, "LFC")
LFC = pad_columns_to_98Q4(LFC, "LFC")
LFC = rows_to_quarter(LFC)

# Civilian Participation Rate, 16+, Constructed (LFPART)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/lfpart
LFPART = pd.read_excel("./data/project data/lfpartMvMd.xlsx", index_col="DATE").diff()
LFPART = month_to_quarter(LFPART, "LFPART")
LFPART = pad_columns_to_98Q4(LFPART, "LFPART")
LFPART = rows_to_quarter(LFPART)

# Unemployment Rate (RUC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ruc
RUC = pd.read_excel("./data/project data/rucQvMd.xlsx", index_col="DATE").diff().fillna(-999)
RUC = rows_to_quarter(RUC)

# Nonfarm Payroll Employment (EMPLOY)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/employ
EMPLOY = pd.read_excel("./data/project data/employMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
EMPLOY = month_to_quarter(EMPLOY, "EMPLOY")
EMPLOY.drop(columns=["EMPLOY65Q1", "EMPLOY65Q2", "EMPLOY65Q3"], inplace=True)
EMPLOY.fillna(-999, inplace=True)
EMPLOY = rows_to_quarter(EMPLOY)

# Indexes of Aggregate Weekly Hours: Total (H)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/h
H = pd.read_excel("./data/project data/hMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
H = month_to_quarter(H, "H")
na_cols_H = ["H65Q4"]
na_cols_H.extend([f'H{i}Q{j}' for i in range(66, 71) for j in range (1, 5)])
na_cols_H.extend(["H71Q1", "H71Q2", "H71Q3"])
na_df_H = pd.DataFrame(columns=na_cols_H)
H = pd.concat([na_df_H, H], axis=1)
H.index.names = ["DATE"]
H.fillna(-999, inplace=True)
H = rows_to_quarter(H)

# Indexes of Aggregate Weekly Hours: Goods Sector (HG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hg
HG = pd.read_excel("./data/project data/hgMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
HG = month_to_quarter(HG, "HG")
na_cols_HG = ["HG65Q4"]
na_cols_HG.extend([f'HG{i}Q{j}' for i in range(66, 71) for j in range (1, 5)])
na_cols_HG.extend(["HG71Q1", "HG71Q2", "HG71Q3"])
na_df_HG = pd.DataFrame(columns=na_cols_HG)
HG = pd.concat([na_df_HG, HG], axis=1)
HG.index.names = ["DATE"]
HG.fillna(-999, inplace=True)
HG = rows_to_quarter(HG)

# Indexes of Aggregate Weekly Hours: Service Sector (HS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hs
HS = pd.read_excel("./data/project data/hsMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
HS = month_to_quarter(HS, "HS")
na_cols_HS = ["HS65Q4"]
na_cols_HS.extend([f'HS{i}Q{j}' for i in range(66, 71) for j in range (1, 5)])
na_cols_HS.extend(["HS71Q1", "HS71Q2", "HS71Q3"])
na_df_HS = pd.DataFrame(columns=na_cols_HS)
HS = pd.concat([na_df_HS, HS], axis=1)
HS.index.names = ["DATE"]
HS.fillna(-999, inplace=True)
HS = rows_to_quarter(HS)

# endregion

# region Labour Productivity and Costs

# Output Per Hour: Business Sector (OPH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/oph
OPH = pd.read_excel("./data/project data/OPHQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
OPH = pad_columns_to_98Q4(OPH, "OPH")

# Unit Labor Costs: Business Sector (ULC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ulc
ULC = pd.read_excel("./data/project data/ULCQvQd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
ULC = pad_columns_to_98Q4(ULC, "ULC")

# endregion

# region Industrial Production & Capacity Utilisation

# Industrial Production Index: Total (IPT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipt
IPT = pd.read_excel("./data/project data/iptMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
IPT = month_to_quarter(IPT, "IPT")
IPT.drop(columns=["IPT62Q4",
                  "IPT63Q1", "IPT63Q2", "IPT63Q3", "IPT63Q4",
                  "IPT64Q1", "IPT64Q2", "IPT64Q3", "IPT64Q4",
                  "IPT65Q1", "IPT65Q2", "IPT65Q3"], inplace=True)
IPT.fillna(-999, inplace=True)
IPT = rows_to_quarter(IPT)

# Industrial Production Index: Manufacturing (IPM)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipm
IPM = pd.read_excel("./data/project data/ipmMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
IPM = month_to_quarter(IPM, "IPM")
IPM.drop(columns=["IPM62Q4",
                  "IPM63Q1", "IPM63Q2", "IPM63Q3", "IPM63Q4",
                  "IPM64Q1", "IPM64Q2", "IPM64Q3", "IPM64Q4",
                  "IPM65Q1", "IPM65Q2", "IPM65Q3"], inplace=True)
IPM.fillna(-999, inplace=True)
IPM = rows_to_quarter(IPM)

# Capacity Utilization Rate: Total (CUT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cut
CUT = pd.read_excel("./data/project data/cutMvMd.xlsx", index_col="DATE")
na_rows_CUT = pd.DataFrame(columns=CUT.columns, index=["1947:02", "1947:05", "1947:08", "1947:11"])
CUT = pd.concat([na_rows_CUT, CUT], axis=0)
CUT = month_to_quarter(CUT, "CUT")
na_cols_CUT = ["CUT65Q4"]
na_cols_CUT.extend([f'CUT{i}Q{j}' for i in range(66, 83) for j in range(1, 5)])
na_cols_CUT.extend(["CUT83Q1", "CUT83Q2"])
na_df_CUT = pd.DataFrame(columns=na_cols_CUT)
CUT = pd.concat([na_df_CUT, CUT], axis=1)
CUT.index.names = ["DATE"]
CUT.fillna(-999, inplace=True)
CUT = rows_to_quarter(CUT)

# Capacity Utilization Rate: Manufacturing (CUM)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cum
CUM = pd.read_excel("./data/project data/cumMvMd.xlsx", index_col="DATE")
na_rows_CUM = pd.DataFrame(columns=CUM.columns, index=["1947:02", "1947:05", "1947:08", "1947:11"])
CUM = pd.concat([na_rows_CUM, CUM], axis=0)
CUM = month_to_quarter(CUM, "CUM")
na_cols_CUM = ["CUM65Q4"]
na_cols_CUM.extend([f'CUM{i}Q{j}' for i in range(66, 79) for j in range(1, 5)])
na_cols_CUM.extend(["CUM79Q1", "CUM79Q2"])
na_df_CUM = pd.DataFrame(columns=na_cols_CUM)
CUM = pd.concat([na_df_CUM, CUM], axis=1)
CUM.index.names = ["DATE"]
CUM.fillna(-999, inplace=True)
CUM = rows_to_quarter(CUM)

# endregion

# region Housing

# Housing Starts (HSTARTS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hstarts
HSTARTS = pd.read_excel("./data/project data/hstartsMvMd.xlsx", index_col="DATE").apply(lambda x: np.log(x)).diff()
HSTARTS = month_to_quarter(HSTARTS, "HSTARTS")
na_cols_HSTARTS = ["HSTARTS65Q4",
                   "HSTARTS66Q1", "HSTARTS66Q2", "HSTARTS66Q3", "HSTARTS66Q4",
                   "HSTARTS67Q1", "HSTARTS67Q2", "HSTARTS67Q3", "HSTARTS67Q4"]
na_df_HSTARTS = pd.DataFrame(columns=na_cols_HSTARTS)
HSTARTS = pd.concat([na_df_HSTARTS, HSTARTS], axis=1)
HSTARTS.index.names = ["DATE"]
HSTARTS.fillna(-999, inplace=True)
HSTARTS = rows_to_quarter(HSTARTS)

# Real Gross Private Domestic Investment: Residential (RINVRESID)
# Same as above
 
# endregion

# Real GNP/GDP (ROUTPUT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/routput
ROUTPUT = pd.read_excel("./data/project data/ROUTPUTQvQd.xlsx", index_col="DATE").diff().fillna(-999)

macro_variables = [RCON, rcong, RCONND, RCOND, RCONS, rconshh, rconsnp, rinvbf, rinvresid,
                   rinvchi, RNX, REX, RIMP, RG, RGF, RGSL, rconhh, WSD, OLI, PROPI, RENTI,
                   DIV, PINTI, TRANR, SSCONTRIB, NPI, PTAX, NDPI, NCON, PINTPAID, TRANPF,
                   NPSAV, RATESAV, NCPROFAT, NCPROFATW, M1, M2, CPI, PCPIX, PPPI, PPPIX,
                   P, PCON, pcong, pconshh, pconsnp, pconhh, PCONX, PIMP, POP, LFC, LFPART,
                   RUC, EMPLOY, H, HG, HS, OPH, ULC, IPT, IPM, CUT, CUM, HSTARTS, ROUTPUT]

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(macro_variables, f)