# =============================================================================
# The idea of this file: additional visualization for the purpose of exploration 
# author: P. Barli
# =============================================================================

# load libraries 
import matplotlib.pyplot as plt

#load functions
from core_v4      import mod1c_g4
import numpy as np

#choosing right modules and stuff happens in settings
import settings

#set up model environment
runD = mod1c_g4(settings.constants, settings.phys_pars, settings.geo_pars, settings.forc_pars)

# plot the domain 
x = runD.px
y_top = runD.b / 2 
y_bot = -runD.b / 2 

plt.figure(figsize=(10,4))
plt.plot(x, y_top, 'k')
plt.plot(x, y_bot, 'k')
plt.fill_between(x, y_bot, y_top, color='lightblue')
plt.xlabel("x (km)")
plt.ylabel("y (m)")
plt.title("Plan view of estuary")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()