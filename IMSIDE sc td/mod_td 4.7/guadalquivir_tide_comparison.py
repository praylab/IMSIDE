# Compare simulated vs observed salinity in the Guadalquivir estuary, and the
# computational cost, for three modelling approaches:
#   1) resolving the tide
#   2) no tide, constant horizontal diffusivity with Kh = 42
#   3) no tide, velocity-dependent ('cub') horizontal diffusivity with ch_st = 150e-3
# Script written by P. Barli, 2026-07-01
#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_v4 import mod1c_g4
from inpu.load_geo_v4 import geo_GUA2
from inpu.load_phys_v4 import phys_gen, phys_GUA_notide
from inpu.load_forc_td_v4 import forc_GUA_Q

# =============================================================================
# Helper functions
# =============================================================================

def calc_s(self, output):
    '''
    Function to calculate subtidal salinity time series from the model output
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

# calculate tidal flow 
def calc_utx(model): 
    '''
    function to calculate along estuary tidal current amplitude
    '''
    # compose a tidal set
    tid_set = {
        'tid_comp': model.tid_comp[0], 
        'tid_per': model.tid_per[0], 
        'a_tide': model.a_tide[0], 
        'p_tide': model.p_tide[0], 
        'Av_ti': model.Av_ti[0], 
        'Kv_ti': model.Kv_ti[0], 
        'sf_ti': model.sf_ti[0], 
        'rr_ti': model.rr_ti[0], 
        'Kh_ti': model.Kh_ti[0], 
        'cv_ti': model.cv_ti[0], 
        'Sc_ti': model.Sc_ti[0], 
    }

    # calculate tidal flow 
    tidal = model.tidal_module(tid_set)

    # calculate 1 tidal cycle 
    t = np.linspace(0, 2*np.pi/tidal['omega'], 200)
    u_signal = [np.real(u * np.exp(1j * tidal['omega'] * t)) for u in tidal['utb']]
    ut_x = np.max(u_signal, axis=1)

    return ut_x


def run_model(constants, phys_pars, geo_pars, forc_pars, xlocs_obs, zlocs_obs, tidal, ux):
    '''
    Run the model and extract the salinity time series at the two observation stations.
    Returns s1, s2 and the wall-clock time (s) spent in solve_eqs.
    '''
    # set up model environment
    run = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)

    # calculate ux 
    if ux == 'vary': 
        run.Ut_x = calc_utx(run)
    elif ux == 'constant': 
        run.Ut_x = None

    # solve equations, timing only the solve step
    t0 = time.time()
    out = run.solve_eqs('D', tidal)
    comp_time = time.time() - t0

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

    # select salinity along the estuary in the middle of the simulation 
    s_est = s_all[len(s_all)//2, :, :]

    return s1, s2, s_est, comp_time


def agreement(sim, obs):
    '''
    RMSE and Pearson correlation between simulation and observation, ignoring NaNs
    '''
    mask = np.isfinite(sim) & np.isfinite(obs)
    rmse = np.sqrt(np.mean((sim[mask]-obs[mask])**2))
    corr = np.corrcoef(sim[mask], obs[mask])[0,1]
    return rmse, corr

# =============================================================================
# load river discharge and salinity observations, summer 2009
# =============================================================================
data_path = 'C:/Users/pba003/Documents/IMSIDE/data/biemond2024/'
Q_GUA = np.loadtxt(data_path + 'Q_Guadalquivir2009.txt')
t_GUA = np.array(np.loadtxt(data_path + 't_Guadalquivir2009.txt', dtype = str),dtype = np.datetime64())

s1o_GUA = np.loadtxt(data_path + 's_Guadalquivir2009_stat1.txt')
s2o_GUA = np.loadtxt(data_path + 's_Guadalquivir2009_stat2.txt')

xlocs = pd.read_excel(data_path + 'locations_saltobs.xlsx', sheet_name = 'longitudinal')
zlocs = pd.read_excel(data_path + 'locations_saltobs.xlsx', sheet_name = 'vertical')

xobs_GUA = xlocs['Guadalquivir']
zobs_GUA = zlocs['Guadalquivir']

constants = phys_gen()
geo_pars_GUA = geo_GUA2()

#%%
# =============================================================================
# define the three cases to compare
# =============================================================================
# note: fully removing the tidal component (tid_comp = []) leaves the Newton-Raphson
# system singular (dgstrf info 7 -> NaN salinity) partway through the series, so "no
# tide" is instead realised by keeping the M2 boundary condition but setting a_tide = 0
# (forc_GUA_Q22), which is numerically stable and physically equivalent (zero tidal forcing).
cases = {
    'tide'        : {'label': 'Tide-resolving',            'phys': phys_GUA_notide(Kh_st_t=22, ch_st_t=None, choice_diffusivityh_st_t='constant'),          'forc': forc_GUA_Q(Q_GUA),      'out_suffix': 'GUA', 'tidal': True, 'ux': 'constant'},
    'notide_Kh42' : {'label': 'No tide, Kh=100 (constant)',  'phys': phys_GUA_notide(Kh_st_t=100, ch_st_t=None, choice_diffusivityh_st_t='constant'),'forc': forc_GUA_Q(Q_GUA), 'out_suffix': 'GUA_notide_kh', 'tidal': False, 'ux': 'constant'},
    'notide_cub'  : {'label': 'No tide, cub (ch_st=150e-3)','phys': phys_GUA_notide(Kh_st_t=None, ch_st_t=150e-3, choice_diffusivityh_st_t='cub'),    'forc': forc_GUA_Q(Q_GUA), 'out_suffix': 'GUA_notide_cub', 'tidal': False, 'ux': 'constant'},
    'notide_ux'  : {'label': 'No tide, ux (ch_st=150e-3)','phys': phys_GUA_notide(Kh_st_t=None, ch_st_t=150e-3, choice_diffusivityh_st_t='cub'),    'forc': forc_GUA_Q(Q_GUA), 'out_suffix': 'GUA_notide_ux', 'tidal': False, 'ux': 'vary'},
}

#%%
results = {}
s_estall = {}
for key, case in cases.items():
    print('Running case: ', case['label'])

    s1, s2, s_est, comp_time = run_model(constants, case['phys'], geo_pars_GUA, case['forc'], xobs_GUA, zobs_GUA, tidal=case['tidal'], ux=case['ux'])
    results[key] = {'s1': s1, 's2': s2, 'time': comp_time}
    s_estall[key] = s_est
    print('  took ', comp_time, ' seconds')

    np.savetxt(data_path + 'output/s1m_' + case['out_suffix'] + '.txt', s1)
    np.savetxt(data_path + 'output/s2m_' + case['out_suffix'] + '.txt', s2)

# =============================================================================
# agreement metrics and computation time summary
# =============================================================================
summary_lines = []
summary_lines.append(f"{'case':30s} {'time [s]':>10s} {'RMSE s1':>10s} {'corr s1':>10s} {'RMSE s2':>10s} {'corr s2':>10s}")
for key, case in cases.items():
    r = results[key]
    rmse1, corr1 = agreement(r['s1'], s1o_GUA)
    rmse2, corr2 = agreement(r['s2'], s2o_GUA)
    results[key]['rmse1'], results[key]['corr1'] = rmse1, corr1
    results[key]['rmse2'], results[key]['corr2'] = rmse2, corr2
    summary_lines.append(f"{case['label']:30s} {r['time']:10.2f} {rmse1:10.3f} {corr1:10.3f} {rmse2:10.3f} {corr2:10.3f}")

summary_text = '\n'.join(summary_lines)
print(summary_text)

with open(data_path + 'output/comparison_summary.txt', 'w') as f:
    f.write(summary_text)

# =============================================================================
# plot the results
# =============================================================================
#%%
fig, axs = plt.subplots(4,1, figsize = (10,9), gridspec_kw={'height_ratios':[1,1.3,1.3,0.8]})

colors = {'tide': 'red', 'notide_Kh42': 'green', 'notide_cub': 'purple', 'notide_ux': 'orange'}

# discharge
axs[0].plot(t_GUA, Q_GUA, c='black')
axs[0].set_ylabel('$Q$ [m$^3$s$^{-1}$]')
axs[0].set_xticklabels([])

# station 1
axs[1].plot(t_GUA, s1o_GUA, c='blue', label='obs')
for key, case in cases.items():
    axs[1].plot(t_GUA, results[key]['s1'], c=colors[key], linestyle='--', label=case['label'])
axs[1].set_ylabel('$s$ [g kg$^{-1}$] (station 1)')
axs[1].legend(fontsize=8)
axs[1].set_xticklabels([])

# station 2
axs[2].plot(t_GUA, s2o_GUA, c='blue', label='obs')
for key, case in cases.items():
    axs[2].plot(t_GUA, results[key]['s2'], c=colors[key], linestyle='--', label=case['label'])
axs[2].set_ylabel('$s$ [g kg$^{-1}$] (station 2)')
axs[2].tick_params(axis='x', rotation=90)

# computation time bar chart
labels = [case['label'] for case in cases.values()]
times = [results[key]['time'] for key in cases.keys()]
axs[3].barh(labels, times, color=[colors[key] for key in cases.keys()])
axs[3].set_xlabel('computation time [s]')

plt.tight_layout()
name_fig = 'tide_comparison_v2.png'
# plt.savefig('C:/Users/pba003/Documents/IMSIDE/figs/' + name_fig, dpi=600, bbox_inches='tight')
plt.show()
# %%
