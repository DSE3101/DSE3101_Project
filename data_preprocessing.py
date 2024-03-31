import pandas as pd
# Data downloaded as of 31 Mar 2024
# Focus on quarterly real data
# Excluded discontinued variables

# region NIPA Product Side - Real

# Real Personal Consumption Expenditures: Total (RCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcon
rcon = pd.read_excel("./data/project data/RCONQvQd.xlsx")

# Real Personal Consumption Expenditures: Goods (RCONG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcong
rcong = pd.read_excel("./data/project data/rcongQvQd.xlsx")

# Real Personal Consumption Expenditures: Nondurable Goods (RCONND)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconnd
rconnd = pd.read_excel("./data/project data/RCONNDQvQd.xlsx")

# Real Personal Consumption Expenditures: Durable Goods (RCOND)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcond
rcond = pd.read_excel("./data/project data/RCONDQvQd.xlsx")

# Real Personal Consumption Expenditures: Services (RCONS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rcons
rcons = pd.read_excel("./data/project data/RCONSQvQd.xlsx")

# Real Household Consumption Expenditures for Services (RCONSHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconshh
rconshh = pd.read_excel("./data/project data/rconshhQvQd.xlsx")

# Real Final Consumption Expenditures of NPISH (RCONSNP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconsnp
rconsnp = pd.read_excel("./data/project data/rconsnpQvQd.xlsx")

# Real Gross Private Domestic Investment: Nonresidential (RINVBF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvbf
rinvbf = pd.read_excel("./data/project data/rinvbfQvQd.xlsx")

# Real Gross Private Domestic Investment: Residential (RINVRESID)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvresid
rinvresid = pd.read_excel("./data/project data/rinvresidQvQd.xlsx")

# Real Gross Private Domestic Investment: Change in Private Inventories (RINVCHI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rinvchi
rinvchi = pd.read_excel("./data/project data/rinvchiQvQd.xlsx")

# Real Net Exports of Goods and Services (RNX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rnx
rnx = pd.read_excel("./data/project data/RNXQvQd.xlsx")

# Real Exports of Goods and Services (REX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rex
rex = pd.read_excel("./data/project data/REXQvQd.xlsx")

# Real Imports of Goods and Services (RIMP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rimp
rimp = pd.read_excel("./data/project data/RIMPQvQd.xlsx")

# Real Government Consumption & Gross Investment: Total (RG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rg
rg = pd.read_excel("./data/project data/RGQvQd.xlsx")

# Real Government Consumption & Gross Investment: Federal (RGF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rgf
rgf = pd.read_excel("./data/project data/RGFQvQd.xlsx")

# Real Government Consumption & Gross Investment: State and Local (RGSL)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rgsl
rgsl = pd.read_excel("./data/project data/RGSLQvQd.xlsx")

# endregion

# region NIPA By Major Function - Real

# Real Household Consumption Expenditures (RCONHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/rconhh
rconhh = pd.read_excel("./data/project data/rconhhQvQd.xlsx")

# Real Final Consumption Expenditures of NPISH (RCONSNP)
# Same as above

# endregion

# region NIPA Personal Income Side

# Wage and Salary Disbursements (WSD)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/wsd
wsd = pd.read_excel("./data/project data/wsdQvQd.xlsx")

# Other Labor Income (OLI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/oli
oli = pd.read_excel("./data/project data/oliQvQd.xlsx")

# Proprietors' Income (PROPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/propi
propi = pd.read_excel("./data/project data/propiQvQd.xlsx")

# Rental Income of Persons (RENTI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/renti
renti = pd.read_excel("./data/project data/rentiQvQd.xlsx")

# Dividends (DIV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/div
div = pd.read_excel("./data/project data/divQvQd.xlsx")

# Personal Interest Income (PINTI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pinti
pinti = pd.read_excel("./data/project data/pintiQvQd.xlsx")

# Transfer Payments (TRANR)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/tranr
tranr = pd.read_excel("./data/project data/tranrQvQd.xlsx")

# Personal Contributions for Social Insurance (SSCONTRIB)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/sscontrib
sscontrib = pd.read_excel("./data/project data/sscontribQvQd.xlsx")

# Nominal Personal Income (NPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/npi
npi = pd.read_excel("./data/project data/npiQvQd.xlsx")

# Personal Tax & Nontax Payments (PTAX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ptax
ptax = pd.read_excel("./data/project data/ptaxQvQd.xlsx")

# Nominal Disposable Personal Income (NDPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ndpi
ndpi = pd.read_excel("./data/project data/ndpiQvQd.xlsx")

# Nominal Personal Consumption Expenditures (NCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncon
ncon = pd.read_excel("./data/project data/nconQvQd.xlsx")

# Interest Paid by Consumers (PINTPAID)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pintpaid
pintpaid = pd.read_excel("./data/project data/pintpaidQvQd.xlsx")

# Personal Transfer Payments to Foreigners (TRANPF)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/tranpf
tranpf = pd.read_excel("./data/project data/tranpfQvQd.xlsx")

# Nominal Personal Saving (NPSAV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/npsav
npsav = pd.read_excel("./data/project data/npsavQvQd.xlsx")

# Personal Saving Rate, Constructed (RATESAV)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ratesav
ratesav = pd.read_excel("./data/project data/ratesavQvQd.xlsx")

# endregion

# region NIPA Other Income

# Nominal Corporate Profits After Tax Without IVA/CCAdj (NCPROFAT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncprofat
ncprofat = pd.read_excel("./data/project data/NCPROFATQvQd.xlsx")

# Nominal Corporate Profits After Tax With IVA/CCAdj (NCPROFATW)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ncprofatw
ncprofatw = pd.read_excel("./data/project data/NCPROFATWQvQd.xlsx")

# endregion

# region Monetary and Financial

# M1 Money Stock (M1)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/m1
m1 = pd.read_excel("./data/project data/m1QvMd.xlsx")

# M2 Money Stock (M2)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/m2
m2 = pd.read_excel("./data/project data/m2QvMd.xlsx")

# endregion

# region Price Level Indices

# Consumer Price Index, Quarterly Vintages (CPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cpi
cpi = pd.read_excel("./data/project data/cpiQvMd.xlsx")

# Core Consumer Price Index (PCPIX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpix
pcpix = pd.read_excel("./data/project data/pcpixMvMd.xlsx")

# Producer Price Index, Final Demand Finished Goods (PPPI)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pppi
pppi = pd.read_excel("./data/project data/pppiMvMd.xlsx")

# Core Producer Price Index, Final Demand Finished Goods (PPPIX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pppix
pppix = pd.read_excel("./data/project data/pppixMvMd.xlsx")

# Price Index for GNP/GDP (P)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/p
p = pd.read_excel("./data/project data/PQvQd.xlsx")

# Price Index for Personal Consumption Expenditures, Constructed (PCON)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcon
pcon = pd.read_excel("./data/project data/pconQvQd.xlsx")

# Price Index for Personal Consumption Expenditures: Goods (PCONG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcong
pcong = pd.read_excel("./data/project data/pcongQvQd.xlsx")

# Price Index for Household Consumption Expenditures for Services (PCONSHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconshh
pconshh = pd.read_excel("./data/project data/pconshhQvQd.xlsx")

# Price Index for Final Consumption Expenditures of NPISH (PCONSNP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconsnp
pconsnp = pd.read_excel("./data/project data/pconsnpQvQd.xlsx")

# Price Index for Household Consumption Expenditures (PCONHH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconhh
pconhh = pd.read_excel("./data/project data/pconhhQvQd.xlsx")

# Core Price Index for Personal Consumption Expenditures (PCONX)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pconx
pconx = pd.read_excel("./data/project data/PCONXQvQd.xlsx")

# Price Index for Imports of Goods and Services (PIMP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pimp
pimp = pd.read_excel("./data/project data/pimpQvQd.xlsx")

# endregion

# region Labour Market

# Civilian Noninstitutional Population, 16+ (POP)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pop
pop = pd.read_excel("./data/project data/popMvMd.xlsx")

# Civilian Labor Force, 16+ (LFC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/lfc
lfc = pd.read_excel("./data/project data/lfcMvMd.xlsx")

# Civilian Participation Rate, 16+, Constructed (LFPART)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/lfpart
lfpart = pd.read_excel("./data/project data/lfpartMvMd.xlsx")

# Unemployment Rate (RUC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ruc
ruc = pd.read_excel("./data/project data/rucQvMd.xlsx")

# Nonfarm Payroll Employment (EMPLOY)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/employ
employ = pd.read_excel("./data/project data/employMvMd.xlsx")

# Indexes of Aggregate Weekly Hours: Total (H)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/h
h = pd.read_excel("./data/project data/hMvMd.xlsx")

# Indexes of Aggregate Weekly Hours: Goods Sector (HG)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hg
hg = pd.read_excel("./data/project data/hgMvMd.xlsx")

# Indexes of Aggregate Weekly Hours: Service Sector (HS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hs
hs = pd.read_excel("./data/project data/hsMvMd.xlsx")

# endregion

# region Labour Productivity and Costs

# Output Per Hour: Business Sector (OPH)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/oph
####################################################################################### Vintage 1998:Q4 to present
oph = pd.read_excel("./data/project data/OPHQvQd.xlsx")

# Unit Labor Costs: Business Sector (ULC)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ulc
####################################################################################### Vintage 1998:Q4 to present
ulc = pd.read_excel("./data/project data/ULCQvQd.xlsx")

# endregion

# region Industrial Production & Capacity Utilisation
# Industrial Production Index: Total (IPT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipt
####################################################################################### Vintage 1962:M11 to 2024:M3
####################################################################################### A LOT OF DATA MISSING
####################################################################################### Data monthly 1919:01 to 2024:02
ipt = pd.read_excel("./data/project data/iptMvMd.xlsx")

# Industrial Production Index: Manufacturing (IPM)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipm
####################################################################################### Vintage 1962:M11 to 2024:M3
####################################################################################### A LOT OF DATA MISSING
####################################################################################### Data monthly 1919:01 to 2024:02
ipm = pd.read_excel("./data/project data/ipmMvMd.xlsx")

# Capacity Utilization Rate: Total (CUT)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cut
####################################################################################### Vintage 1983:M7 to 2024:M3
####################################################################################### A LOT OF DATA MISSING
####################################################################################### Data monthly 1948:01 to 2024:02
cut = pd.read_excel("./data/project data/cutMvMd.xlsx")

# Capacity Utilization Rate: Manufacturing (CUM)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cum
####################################################################################### Vintage 1979:M8 to 2024:M3
####################################################################################### Data monthly 1948:01 to 2024:02
cum = pd.read_excel("./data/project data/cumMvMd.xlsx")

print(ipt)
print(ipm)
print(cut)
print(cum)
# endregion

# region Housing
# Housing Starts (HSTARTS)
# https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/hstarts
####################################################################################### Vintage 1968:M2 to 2024:M3
####################################################################################### A LOT OF DATA MISSING
####################################################################################### Data monthly 1947:01 to 2024:02
hstarts = pd.read_excel("./data/project data/hstartsMvMd.xlsx")
print(hstarts)
# Real Gross Private Domestic Investment: Residential (RINVRESID)
# Same as above
# endregion