# calculate tidal current

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
# helper function 
# =============================================================================
def tidal_module(self, tid_set):
    # =============================================================================
    # this function calculates the quantities which are required for futher calculations
    # moslty solves for the tidal water level and associated currents
    # =============================================================================

    #define parameters
    #vertical visocisty
    if self.choice_viscosityv_ti == 'constant': 0 #do nothing, value is specified
    elif self.choice_viscosityv_ti == 'cuh': tid_set['Av_ti'] = tid_set['cv_ti'] * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity tidal')

    #vertical diffusivity
    if self.choice_diffusivityv_ti == 'constant': 0 #do nothing, value is specified
    elif self.choice_diffusivityv_ti == 'as': tid_set['Kv_ti'] = tid_set['Av_ti'] / tid_set['Sc_ti']
    else: print('ERROR: no valid option for choice vertical diffusivity tidal')

    #bottom slip 
    if self.choice_bottomslip_ti  == 'constant' : 0 #do nothing, value is specified
    elif self.choice_bottomslip_ti== 'rr' : tid_set['sf_ti'] = tid_set['Av_ti'] / (tid_set['rr_ti'] * self.H) 
    else: print('ERROR: invalid choice of tidal bottom slip')

    # =============================================================================
    # solve for water level in different segments (except for the sea domain)
    # matrix equation, as in fork
    # use Jinyang notation here - bit weird but ok
    # =============================================================================
    
    # derivative, set sea domain with av in estuary mouth and limit to av_min
    av_x = np.zeros(tid_set['Av_ti'].shape) + tid_set['Av_ti']
    av_x[self.di[-2]+1:self.di[-1]] = tid_set['Av_ti'][self.di[-2]]

    avti_min = 0.0010 # arbitrary number 
    av_x = np.clip(av_x, a_min=avti_min, a_max=None)
        
    #parameters for equations
    omega = 2*np.pi/tid_set['tid_per']
    deA = (1+1j)*self.H/np.sqrt(2*av_x/omega)
    B = (np.cosh(deA) + av_x/(tid_set['sf_ti']*self.H) * deA * np.sinh(deA))**-1
    ka = np.sqrt(0.25*self.bex**-2 + omega**2/(self.g*self.H) * (B/deA*np.sinh(deA)-1)**-1)

    sol = np.zeros(2*len(self.nxn)+1,dtype=complex)
    matr = np.zeros((2*len(self.nxn)+1,2*len(self.nxn)+1),dtype=complex)
    
    #river boundary
    matr[0,0] = -1/(2*self.bn[0]) + ka[0]
    matr[0,1] = -1/(2*self.bn[0]) - ka[0]
    sol[0] = 0
        
    fQ = self.b * self.H * (B/deA * np.sinh(deA) - 1)
    
    for dom in range(len(self.nxn)-1): 
        #water level equal
        matr[dom*2+1, dom*2+0] = np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(self.Ln[dom]*ka[self.di[dom+1]-1])
        matr[dom*2+1, dom*2+1] = np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(-self.Ln[dom]*ka[self.di[dom+1]-1])
        matr[dom*2+1, dom*2+2] = -1
        matr[dom*2+1, dom*2+3] = -1
        #discharge equal, i.e. water level gradient
        matr[dom*2+2, dom*2+0] = (np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp( self.Ln[dom]*ka[self.di[dom+1]-1]) * (-1/(2*self.bn[dom]) + ka[self.di[dom+1]-1])) * fQ[self.di[dom+1]-1]
        matr[dom*2+2, dom*2+1] = (np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(-self.Ln[dom]*ka[self.di[dom+1]-1]) * (-1/(2*self.bn[dom]) - ka[self.di[dom+1]-1])) * fQ[self.di[dom+1]-1]
        matr[dom*2+2, dom*2+2] = -(-1/(2*self.bn[dom+1]) + ka[self.di[dom+1]]) * fQ[self.di[dom+1]]
        matr[dom*2+2, dom*2+3] = -(-1/(2*self.bn[dom+1]) - ka[self.di[dom+1]]) * fQ[self.di[dom+1]]
    
        #solution vector is zero everywhere, no need to specify that
        #print(self.di[dom+1]-1, self.di[dom+1])
    
    #sea boundary
    
    #first condition: at the sea boundary, the level is equal to a to be determined level
    matr[-2,-3] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp( ka[-1]*self.Ln[-1])
    matr[-2,-2] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp(-ka[-1]*self.Ln[-1])
    matr[-2,-1] = -1
    
    #second condition: the water level difference in the sea domain equals the difference between the prescribed sea level and the to be determined level
    matr[-1,-3] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp( ka[-1]*self.Ln[-1]) - 1
    matr[-1,-2] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp(-ka[-1]*self.Ln[-1]) - 1
    matr[-1,-1] = -1 
    sol[-1] = tid_set['a_tide'] * np.exp(-1j*tid_set['p_tide']/180*np.pi)
    
    #solve this set of equations
    oplossing = np.linalg.solve(matr,sol)
    coef_eta = oplossing[:-1].reshape((int(len(self.nxn)),2))
       
    #print(oplossing)

    # =============================================================================
    # calculate water level and derivatives
    # =============================================================================    
    eta     = np.zeros(self.di[-1],dtype=complex)
    etar    = np.zeros(self.di[-1],dtype=complex)
    detadx  = np.zeros(self.di[-1],dtype=complex)
    detadx2 = np.zeros(self.di[-1],dtype=complex)
    detadx3 = np.zeros(self.di[-1],dtype=complex)
    for dom in range(self.ndom):
        x_here = np.linspace(-self.Ln[dom],0,self.nxn[dom])
        eta[self.di[dom]:self.di[dom+1]]  = np.exp(-(x_here+self.Ln[dom])/(2*self.bn[dom])) * (coef_eta[dom,0]*np.exp((x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) + coef_eta[dom,1]*np.exp(-(x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) )
        etar[self.di[dom]:self.di[dom+1]] = np.exp(-(x_here+self.Ln[dom])/(2*self.bn[dom])) * (coef_eta[dom,0]*np.exp((x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) - coef_eta[dom,1]*np.exp(-(x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) )
        
        detadx[self.di[dom]:self.di[dom+1]]  = ka[self.di[dom]:self.di[dom+1]] * etar[self.di[dom]:self.di[dom+1]] - eta[self.di[dom]:self.di[dom+1]]/(2*self.bn[dom])
        detadx2[self.di[dom]:self.di[dom+1]] = ka[self.di[dom]:self.di[dom+1]]**2*eta[self.di[dom]:self.di[dom+1]] + eta[self.di[dom]:self.di[dom+1]]/(4*self.bn[dom]**2) - ka[self.di[dom]:self.di[dom+1]]/self.bn[dom] * etar[self.di[dom]:self.di[dom+1]]
        detadx3[self.di[dom]:self.di[dom+1]] = ka[self.di[dom]:self.di[dom+1]]**3 * etar[self.di[dom]:self.di[dom+1]] - eta[self.di[dom]:self.di[dom+1]]/(8*self.bn[dom]**3) - 3*eta[self.di[dom]:self.di[dom+1]]*ka[self.di[dom]:self.di[dom+1]]**2/(2*self.bn[dom]) + 3*ka[self.di[dom]:self.di[dom+1]]/(4*self.bn[dom]**2) * etar[self.di[dom]:self.di[dom+1]]
    

    #correct shape 
    eta     = eta[np.newaxis,:,np.newaxis]
    detadx  = detadx[np.newaxis,:,np.newaxis]
    detadx2 = detadx2[np.newaxis,:,np.newaxis]
    detadx3 = detadx3[np.newaxis,:,np.newaxis]
    
    #velocities
    ut    = (self.g/(1j*omega) * detadx  * (B[:,np.newaxis]*np.cosh(deA[:,np.newaxis]*self.z_nd) - 1))[0]
    dutdx = (self.g/(1j*omega) * detadx2 * (B[:,np.newaxis]*np.cosh(deA[:,np.newaxis]*self.z_nd) - 1))[0]
    wt    = 1j*omega*eta - self.g/(1j*omega) * (detadx2 + detadx/self.bex[np.newaxis,:self.di[-1],np.newaxis]) * (B[:,np.newaxis]*self.H[:,np.newaxis]/deA[:,np.newaxis]*np.sinh(deA[:,np.newaxis]*self.z_nd) - self.zlist)

    utb = (self.g/(1j*omega) * detadx[0,:,0] * (B/deA*np.sinh(deA)-1))
    utp = (self.g/(1j*omega) * detadx[0] * B[:,np.newaxis] * (np.cosh(deA[:,np.newaxis]*self.z_nd) - np.sinh(deA[:,np.newaxis])/deA[:,np.newaxis]))
       
    # =============================================================================
    #     save in dictionary
    # =============================================================================
    save = {}
    save['omega'] = omega
    save['deA'] = deA
    save['B']   = B
    
    save['eta']     = eta
    save['detadx']  = detadx
    save['detadx2'] = detadx2
    save['detadx3'] = detadx3
    
    save['ut']    = ut
    save['dutdx'] = dutdx
    save['wt']    = wt
    
    save['utb'] = utb
    save['utp'] = utp
 
    #other 
    save['eps']  = tid_set['Kh_ti']/(omega*self.Lsc**2)       #epsilon, the normalised horizontal diffusion
    save['epsL'] = tid_set['Kh_ti']/(omega)        #epsilon, the normalised horizontal diffusion
    
    #av derivative, sea domain is different 
    save['av_ti'] = av_x
    
    return save  


def calc_avti(ut, model):
    av_x =  model.cv_ti[0] * ut * model.H
    av_x[model.di[-2]+1:model.di[-1]] = av_x[model.di[-2]]

    avti_min = 0.0010 # arbitrary number 
    av_x = np.clip(av_x, a_min=avti_min, a_max=None)

    return av_x


def extract_ut(tidal_flow):
    t = np.linspace(0, 2*np.pi/tidal_flow['omega'], 200)
    u_signal = [np.real(u * np.exp(1j * tidal_flow['omega'] * t)) for u in tidal_flow['utb']]
    ut_x = np.max(u_signal, axis=1)

    return ut_x

def iter_ux(tid_set, alpha, model):
    tol = np.inf
    uts = []
    avs = []

    uts.append(np.zeros(model.di[-1]) + model.Ut)
    avs.append(calc_avti(model.Ut, model))

    while tol > 10e-5:

        tidal_flow = tidal_module(model, tid_set)
        ut_x = extract_ut(tidal_flow)
        av_ti = tidal_flow['av_ti']

        # to avoid the solution to be too far 
        ut_next = alpha * ut_x + (1-alpha) * model.Ut

        # calculate new viscosity param
        av_next = calc_avti(ut_next, model)

        # calculate tolerance
        uts.append(ut_next)
        avs.append(av_next)

        tol = np.max(abs(av_ti - av_next))

        # update ut 
        model.Ut = ut_next

    return tidal_flow, uts, avs

#%%
# ====================================================================
# calculate 
# ====================================================================

# discharge data 
data_path = 'C:/Users/pba003/Documents/IMSIDE/data/biemond2024/'
Q_GUA = np.loadtxt(data_path + 'Q_Guadalquivir2009.txt')
t_GUA = np.array(np.loadtxt(data_path + 't_Guadalquivir2009.txt', dtype = str),dtype = np.datetime64())

# parameters
constants = phys_gen()
phys = phys_GUA_notide(Kh_st_t=None, ch_st_t=150e-3)
forc = forc_GUA_Q(Q_GUA)
geo = geo_GUA2()

# initialise model
gua_model = mod1c_g4(constants, phys, geo, forc)

# set forcing  
tid_set = {
        'tid_comp': gua_model.tid_comp[0], 
        'tid_per': gua_model.tid_per[0], 
        'a_tide': gua_model.a_tide[0], 
        'p_tide': gua_model.p_tide[0], 
        'Av_ti': gua_model.Av_ti[0], 
        'Kv_ti': gua_model.Kv_ti[0], 
        'sf_ti': gua_model.sf_ti[0], 
        'rr_ti': gua_model.rr_ti[0], 
        'Kh_ti': gua_model.Kh_ti[0], 
        'cv_ti': gua_model.cv_ti[0], 
        'Sc_ti': gua_model.Sc_ti[0], 
    }

tidal_flow, uts, avs = iter_ux(tid_set, 0.8, gua_model)

# plot the iteration
colors = plt.cm.viridis(np.linspace(0, 1, len(uts)))

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

for i, (ut_i, av_i) in enumerate(zip(uts, avs)):
    ax[0].plot(gua_model.px, ut_i, color=colors[i])
    ax[1].plot(gua_model.px, av_i, color=colors[i])

ax[0].set_ylabel('Tidal Current Amplitude (m/s)')

ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('vertical viscosity')

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(uts) - 1))
fig.colorbar(sm, ax=ax, label='iteration')

plt.show()

# %%
