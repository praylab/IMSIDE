# This script contain an attempt to replicate Figure 4 of Biemond et al. 2024 (doi: 10.1029/2024JC021294)/
# Script written by P. Barli, 2026-04-21
#%%
# load core model modules 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from core_v4 import mod1c_g4
from inpu.load_geo_v4 import geo_DLW2, geo_GUA2, geo_LOI2
from inpu.load_phys_v4 import phys_gen, phys_DLW2, phys_GUA2, phys_LOI2
from inpu.load_forc_td_v4 import forc_DLW_Q, forc_GUA_Q, forc_LOI_Q

# =============================================================================
# Helper Functions 
# =============================================================================

def calc_s(self, output):
    '''
    Function to calculate subtidal salinty time series from the model output
    '''
    shere = []
    for t in range(self.T):
        ss2 = np.delete(output[t] , np.concatenate([self.ii_all['bnl_rgt'].flatten(),self.ii_all['bnl_lft'].flatten(),self.ii_all['bnl_bnd']]))
        
        s_b = np.reshape(ss2,(self.di[-1],self.N+1))[:,0]
        sn = np.reshape(ss2,(self.di[-1],self.N+1))[:,1:]
        s_p = np.sum([sn[:,n-1]*np.cos(np.pi*n*self.z_nd[:,np.newaxis]) for n in range(1,self.M)] , 0) 
        s = (s_b+s_p)*self.soc_sca
        shere.append(s.T)
        
    shere = np.array(shere)
    return shere

# =============================================================================
# load river discharge and salinity data in the calibration period 
# this study focuses on three estuariesL Delaware, Guadalquivir and Loire
# figure 4 compare the observed and simulated salinity time series in 2 stations
# =============================================================================
data_path = 'C:/Users/pba003/Documents/IMSIDE/data/biemond2024/'
Q_DLW = np.loadtxt(data_path + 'Q_Delaware2023.txt') # summer 2023
Q_GUA = np.loadtxt(data_path + 'Q_Guadalquivir2009.txt') # summer 2009
Q_LOI = np.loadtxt(data_path + 'Q_Loire2013.txt') # summer 2013 

t_DLW = np.array(np.loadtxt(data_path + 't_Delaware2023.txt', dtype = str),dtype = np.datetime64())
t_GUA = np.array(np.loadtxt(data_path + 't_Guadalquivir2009.txt', dtype = str),dtype = np.datetime64())
t_LOI = np.array(np.loadtxt(data_path + 't_Loire2013.txt', dtype = str),dtype = np.datetime64())

s1o_DLW = np.loadtxt(data_path + 's_Delaware2023_stat1.txt')
s1o_GUA = np.loadtxt(data_path + 's_Guadalquivir2009_stat1.txt')
s1o_LOI = np.loadtxt(data_path + 's_Loire2013_stat1.txt')

s2o_DLW = np.loadtxt(data_path + 's_Delaware2023_stat2.txt')
s2o_GUA = np.loadtxt(data_path + 's_Guadalquivir2009_stat2.txt')
s2o_LOI = np.loadtxt(data_path + 's_Loire2013_stat2.txt')

#%%
# load observation locations 
xlocs = pd.read_excel(data_path + 'locations_saltobs.xlsx', sheet_name = 'longitudinal')
zlocs = pd.read_excel(data_path + 'locations_saltobs.xlsx', sheet_name = 'vertical')

def run_model(constants, phys_pars, geo_pars, forc_pars, xlocs_obs, zlocs_obs): 
    '''
    '''
    # set up model environment
    run = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)
    
    # solve equations
    out = run.solve_eqs('D')

    # calculate salinity time series 
    s_all = calc_s(run, out)

    # find the location of the observation stations in the model grid
    pxh = run.px + 25 
    pzh = run.zlist
    xi_1 = np.argmin(np.abs(pxh - xlocs_obs[0]))
    xi_2 = np.argmin(np.abs(pxh - xlocs_obs[1]))

    zi_1 = np.argmin(np.abs(pzh - zlocs_obs[0]))
    zi_2 = np.argmin(np.abs(pzh - zlocs_obs[1]))

    # select the salinity at the observation locations
    s1 = s_all[:,xi_1,zi_1]
    s2 = s_all[:,xi_2,zi_2]

    return s1, s2 

# select starting parameters and observation locations
constants = phys_gen()

# Delaware
phys_pars_DLW = phys_DLW2()
geo_pars_DLW = geo_DLW2()
forc_pars_DLW = forc_DLW_Q(Q_DLW)
xobs_DLW = xlocs['Delaware']
zobs_DLW = zlocs['Delaware']

# Guadalquivir
phys_pars_GUA = phys_GUA2()
geo_pars_GUA = geo_GUA2()
forc_pars_GUA = forc_GUA_Q(Q_GUA)
xobs_GUA = xlocs['Guadalquivir']
zobs_GUA = zlocs['Guadalquivir']

# Loire
phys_pars_LOI = phys_LOI2()
geo_pars_LOI = geo_LOI2()
forc_pars_LOI = forc_LOI_Q(Q_LOI)
xobs_LOI = xlocs['Loire']
zobs_LOI = zlocs['Loire']

s1m_DLW, s2m_DLW = run_model(constants, phys_pars_DLW, geo_pars_DLW, forc_pars_DLW, xobs_DLW, zobs_DLW)
s1m_GUA, s2m_GUA = run_model(constants, phys_pars_GUA, geo_pars_GUA, forc_pars_GUA, xobs_GUA, zobs_GUA)
s1m_LOI, s2m_LOI = run_model(constants, phys_pars_LOI, geo_pars_LOI, forc_pars_LOI, xobs_LOI, zobs_LOI)

np.savetxt(data_path + 'output/s1m_DLW.txt', s1m_DLW)
np.savetxt(data_path + 'output/s2m_DLW.txt', s2m_DLW)

np.savetxt(data_path + 'output/s1m_GUA.txt', s1m_GUA)
np.savetxt(data_path + 'output/s2m_GUA.txt', s2m_GUA)

np.savetxt(data_path + 'output/s1m_LOI.txt', s1m_LOI)
np.savetxt(data_path + 'output/s2m_LOI.txt', s2m_LOI)

# =============================================================================
# Plot the results 
# =============================================================================
#%%load output 
s1m_DLW = np.loadtxt(data_path + 'output/s1m_DLW.txt')
s1m_GUA = np.loadtxt(data_path + 'output/s1m_GUA.txt')
s1m_LOI = np.loadtxt(data_path + 'output/s1m_LOI.txt')

s2m_DLW = np.loadtxt(data_path + 'output/s2m_DLW.txt')
s2m_GUA = np.loadtxt(data_path + 'output/s2m_GUA.txt')
s2m_LOI = np.loadtxt(data_path + 'output/s2m_LOI.txt')

#%% 
fig, axs = plt.subplots(3,3, figsize = (15,10))

# plot the discharge on the first row 
axs[0,0].plot(t_DLW, Q_DLW, c='black')
axs[0,1].plot(t_GUA, Q_GUA, c='black')
axs[0,2].plot(t_LOI, Q_LOI, c='black')

# plot the salinity observed vs simulated at each station 
# Delaware 
axs[1,0].plot(t_DLW, s1o_DLW, c='blue', label = 'obs')
axs[1,0].plot(t_DLW, s1m_DLW, c='red', label = 'sim')

axs[2,0].plot(t_DLW, s2o_DLW, c='blue', label = 'obs')
axs[2,0].plot(t_DLW, s2m_DLW, c='red', label = 'sim')

# Guadalquivir
axs[1,1].plot(t_GUA, s1o_GUA, c='blue', label = 'obs')
axs[1,1].plot(t_GUA, s1m_GUA, c='red', label = 'sim')

axs[2,1].plot(t_GUA, s2o_GUA, c='blue', label = 'obs')
axs[2,1].plot(t_GUA, s2m_GUA, c='red', label = 'sim')

# Loire
axs[1,2].plot(t_LOI, s1o_LOI, c='blue', label = 'obs')
axs[1,2].plot(t_LOI, s1m_LOI, c='red', label = 'sim')

axs[2,2].plot(t_LOI, s2o_LOI, c='blue', label = 'obs')
axs[2,2].plot(t_LOI, s2m_LOI, c='red', label = 'sim')

# layout and labels 
axs[0,0].set_title('Delaware')
axs[0,1].set_title('Guadalquivir')
axs[0,2].set_title('Loire')

axs[1,0].legend()

axs[0,0].set_ylabel('$Q$ [m$^3$s$^{-1}$]')
axs[1,0].set_ylabel('$s$ [g kg$^{-1}$]')
axs[2,0].set_ylabel('$s$ [g kg$^{-1}$]')

for a in range(3):
    axs[0,a].set_xticklabels([]) # remove x-axis ticks in the first row 
    axs[1,a].set_xticklabels([]) # remove x-axis ticks in the second row
    axs[2,a].set_xlabel('Date')
    axs[2,a].tick_params(axis='x', rotation=90)

    axs[a,0].set_xlim(t_DLW[0], t_DLW[-1])
    axs[a,1].set_xlim(t_GUA[0], t_GUA[-1])
    axs[a,2].set_xlim(t_LOI[0], t_LOI[-1])


plt.tight_layout()
name_fig = 'Figure4_Biemond2024.png'
plt.savefig('C:/Users/pba003/Documents/IMSIDE/figs/' + name_fig, dpi=600, bbox_inches='tight')
plt.show()
# %%
