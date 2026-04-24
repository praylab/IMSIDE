# Pre-process data for the freshwater forcing in the IMSIDE model
# author: P. Barli
# March 2026

#%%
import os 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import scipy as sp

#%% 
#=============================================================================
# Build the forcing timeseries based on the discharge from the main river and the tributaries.
# ============================================================================

# load discharge data from main river 
data_folder = 'C:/Users/pba003/Documents/IMSIDE/data/biemond2022/'

data_Q = sp.io.loadmat(os.path.join(data_folder, 'freshwater_discharges.mat'))
Q_gua = np.array(data_Q['Q']).flatten()
t_gua = np.array(data_Q['t']).flatten()

# load dishcharge from tributaries
gergal = np.array(pd.read_excel(os.path.join(data_folder, 'Gergal.xlsx')))
aznalc = np.array(pd.read_excel(os.path.join(data_folder, 'Aznalczar.xlsx')))
guadaira = np.array(pd.read_excel(os.path.join(data_folder, 'Guadaira.xlsx')))
aguila = np.array(pd.read_excel(os.path.join(data_folder, 'TorredelAguila.xlsx')))

# distances from mouth 
d_aznalc = 17.8 #km
d_aguila = 48.1 #km
d_guadaira = 58.9 #km
d_gergal = 91.7 # km

# pre-process tributaries data 

# n_year = len(np.unique(gergal[:,1][1:]))
# year_start = 2008 
# Q_ = np.zeros(n_year*365+2)
# t_ = np.zeros(n_year*365+2) + pd.to_datetime('2007-10-01').value / (1e9*3600*24) + 719529  # convert to days since 1970-01-01
# tdp = 0 

# for i in range(n_year):
#     dim = diml if (year+i)%4==0 else dimn
#     for j in range(12):
#         Q_Ger[tdp+np.sum(dim[:j+1]):tdp+np.sum(dim[:j+2])] = Gergal[1+i*31:1+i*31+dim[j+1] , 3+j]
#     tdp = tdp+np.sum(dim)

## From ChatGPT
import numpy as np
import pandas as pd

def build_daily_series(data, start_year, noy, start_date="2007-10-01",
                       skip_years=None, missing_value=0):

    if skip_years is None:
        skip_years = []

    dimn = [0,31,30,31,31,28,31,30,31,30,31,31,30] #days in month - first month is october here!!
    diml = [0,31,30,31,31,29,31,30,31,30,31,31,30] 

    Q = np.zeros(365*noy+2)
    t = np.arange(365*noy+2) + pd.to_datetime(start_date).value/(10**9*3600*24) + 719529

    tdp = 0
    row_offset = 0

    for i in range(noy):

        year = start_year + i

        # handle skipped years
        if year in skip_years:
            Q[tdp:tdp+365] = missing_value
            tdp += 365
            continue

        dim = diml if year % 4 == 0 else dimn

        for j in range(12):
            Q[
                tdp + np.sum(dim[:j+1]) : tdp + np.sum(dim[:j+2])
            ] = data[
                1 + row_offset*31 : 1 + row_offset*31 + dim[j+1],
                3 + j
            ]

        tdp += np.sum(dim)
        row_offset += 1

    return t, Q

t_Ger, Q_Ger = build_daily_series(
    gergal, 
    start_year=2008,
    noy=5,
)

t_Azn, Q_Azn = build_daily_series(
    aznalc, 
    start_year=2008,
    noy=5,
    skip_years=[2010]
)

t_Gua, Q_Gua = build_daily_series(
    guadaira, 
    start_year=2008,
    noy=5,
    skip_years=[2010, 2011, 2012]
)

t_Agu, Q_Agu = build_daily_series(
    aguila, 
    start_year=2008,
    noy=5,
)

# plt.figure(figsize=(10,6))
# plt.plot(t_Ger, Q_Ger, label='Gergal')
# plt.plot(t_Azn, Q_Azn, label='Aznalczar')
# plt.plot(t_Gua, Q_Gua, label='Guadaira')
# plt.plot(t_Agu, Q_Agu, label='Torre del Aguila')
# plt.xlabel('Time (days since 1970-01-01)')
# plt.ylabel('Discharge (m^3/s)')
# plt.title('Daily Discharge from Tributaries')
# plt.legend()
# plt.show()


#%% discharge from the main river
discharge_data = sp.io.loadmat(data_folder + 'freshwater_discharges.mat')
Q_main = np.array(discharge_data['Q']).flatten()
t_main = np.array(discharge_data['t']).flatten()

def all_Q(t, day_start, n_day):
    Q_main_i = Q_main[np.where(t_main==t)[0][0]+day_start:np.where(t_main==t)[0][0]+n_day+day_start]
    Q_Azn_i = Q_Azn[np.where(t_Azn==t)[0][0]+day_start:np.where(t_Azn==t)[0][0]+n_day+day_start]
    Q_Gua_i = Q_Gua[np.where(t_Gua==t)[0][0]+day_start:np.where(t_Gua==t)[0][0]+n_day+day_start]
    Q_Ger_i = Q_Ger[np.where(t_Ger==t)[0][0]+day_start:np.where(t_Ger==t)[0][0]+n_day+day_start]
    Q_Agu_i = Q_Agu[np.where(t_Agu==t)[0][0]+day_start:np.where(t_Agu==t)[0][0]+n_day+day_start]

    # assuming all the tributaries confluence at the river boundary of the domain 
    Q = Q_main_i + Q_Azn_i + Q_Gua_i + Q_Ger_i + Q_Agu_i

    return Q

ds_2008 = 350 
t2008 = pd.to_datetime('2008-01-01').value/(10**9*3600*24) + 719529
n_day_2008 = 100

# assuming all the tributaries confluence at the river boundary of the domain 
Q_pulse2009 = all_Q(t2008, ds_2008, n_day_2008)

t2009 = pd.to_datetime('2009-01-01').value/(10**9*3600*24) + 719529
ds_2009 = 290
n_day_2009 = 365
Q_pulse2010 = all_Q(t2009, ds_2009, n_day_2009)

np.savetxt(os.path.join(data_folder, 'Q_forcing_pulse2009.txt'), Q_pulse2009)
np.savetxt(os.path.join(data_folder, 'Q_forcing_pulse2010.txt'), Q_pulse2010)

#%% 
# ============================================================================
# Setting up the model 
# ============================================================================
# we will run two cases of freshwater pulse in February 2009 and in 2010
# load the settings and model 
from core_v4 import mod1c_g4

from inpu.load_forc_td_v4 import forc_GUA_Q22
from inpu.load_phys_v4 import phys_gen, phys_GUA2
from inpu.load_geo_v4 import geo_GUA2, geo_GUA_biemond22

# helper function to calculate salt intrusion length (Lint)
def calc_Lint(self, sss, indi):
    '''
    calculate the salt intrusion length as a function of time 
    :param sss: salinity field
    :param indi: indices to map sss to the salinity field 
    '''
    # =============================================================================
    # calculate the salt intrusion length as a function of time 
    # =============================================================================
    Lint = np.zeros(self.T)

    for t in range(self.T):
        ss2 = np.delete(sss[t] , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

        #calculate salt intrusion length 
        sbot = self.soc_sca * (\
            np.reshape(ss2,(self.di[-1],self.M))[:,0] \
                + np.sum(
                    np.reshape(ss2,(self.di[-1],self.M))[:,1:] * np.array([(-1)**n for n in range(1,self.M)]),1
                    )
                    )
        Lint[t] = -self.px[np.where(sbot>2)[0][0]] - self.Ln[-1]/1000 

    return Lint

# No observation data found in open repository 
constants = phys_gen()

# Guadalquivir
phys_pars = phys_GUA2()
geo_pars = geo_GUA_biemond22()
forc_pars = forc_GUA_Q22(Q_pulse2009)

# freshwater pulse in 2009
run2009 = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)
out2009 = run2009.solve_eqs('D')
Lint_2009 = calc_Lint(run2009, out2009, run2009.ii_all)

#%%
# freshwater pulse in 2010 --> could not find the solution  
# forc_pars = forc_GUA_Q22(Q_pulse2010)
# run2010 = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)
# out2010 = run2010.solve_eqs('D')
# Lint_2010 = calc_Lint(out2010, run2010.ii_all)

#%% Plot the result 

# calculate t 
t_pulse2009 = np.arange(len(Q_pulse2009)) + ds_2008 - 365

# plot discharge and salt intrusion length 
fig, ax = plt.subplots(figsize=(6,4))
# plot discharge
ax.plot(t_pulse2009, Q_pulse2009, c='k', label='discharge')
ax.set_ylabel('$Q$ [m$^3$s$^{-1}$]')
ax.set_ylim(0,1000)

# plot salt intrusion length in second y axis 
ax2 = ax.twinx()
ax2.plot(t_pulse2009, Lint_2009, c='dodgerblue', label='Simulated Salt Intrusion Length')
ax2.set_ylabel('$X_2$ [km]')
ax2.set_ylim(5,70)
ax2.tick_params(axis='y', colors='dodgerblue')
ax2.yaxis.label.set_color('dodgerblue')

ax.grid()
ax.set_xlabel('Time [doy 2009]')
plt.tight_layout()
plt.show()

# %%