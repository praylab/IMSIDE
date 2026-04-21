import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp 

disc = sp.io.loadmat('data/freshwater_discharges.mat')

Q_gu = np.array(disc['Q']).flatten()
Qt = np.array(disc['t']).flatten()

dat_start, dat_stop = '2008-10-01' , '2009-03-01'

i_start = np.where(Qt == pd.to_datetime(dat_start+' 00:00:00').value/(10**9*3600*24)+719529)[0]
i_stop =  np.where(Qt == pd.to_datetime(dat_stop+' 00:00:00').value/(10**9*3600*24)+719529)[0]
    
if len(i_start) ==0 or len(i_stop) == 0 : print('ERROR: chosen date not available ')
    
Q_here = Q_gu[i_start[0]:i_stop[0]]
t_here = Qt[i_start[0]:i_stop[0]]

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(t_here, Q_here)

plt.show()


