#load functions
from core_v4      import mod1c_g4
import numpy as np
import matplotlib.pyplot as plt

#choosing right modules and stuff happens in settings
import settings

#set up model environment
model = mod1c_g4(settings.constants, settings.phys_pars, settings.geo_pars, settings.forc_pars)

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
    'cv_ti':model.cv_ti[0], 
    'Sc_ti': model.Sc_ti[0], 
}

# calculate tidal flow 
tidal = model.tidal_module(tid_set)

# calculate 1 tidal cycle 
t = np.linspace(0, 2*np.pi/tidal['omega'], 200)
u_signal = [np.real(u * np.exp(1j * tidal['omega'] * t)) for u in tidal['utb']]
ut_x = np.max(u_signal, axis=1)

# # plot the tidal velocity 
# plt.plot(model.px, ut_x)
# plt.xlabel("t")
# plt.ylabel("maximum tidal velocity (m/s)")
# plt.show()

# check if the current due to river is exceeding the tidal current 
Q_max = 100
u_river = Q_max / (model.H * model.b)

# plot it together 
plt.plot(model.px, ut_x, label='tidal velocity')
plt.plot(model.px, u_river, label='river velocity')
plt.xlabel("x")
plt.ylabel("velocity (m/s)")
plt.legend()
plt.show()

