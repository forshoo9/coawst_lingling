import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from yang_func import *

loc = plticker.MultipleLocator(base=0.25)

low = cm.GnBu_r(np.linspace(0.01,0.9, 128))
mid = np.ones((80-2*30,4))
high = cm.YlOrRd(np.linspace(0.1, 0.95, 128))
colors = np.vstack((low, mid, high))
bwr = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)
bwr.set_over = 'darkbrown'
bwr.set_bad = 'k'

tag = '_50km'
#tag = '_50km_anom'
#tag = '_right_100km'
#tag = '_200km'
#tag = '_right_200km'

date_range = pd.date_range('2019-09-06 00:00:00', '2019-09-08 12:00:00', freq='1H')

atm_mo = xr.open_dataset("storm_center_lingling_case_ATM_MO_1-1-2"+tag+".nc")
atm_mo = atm_mo.assign_coords({"time": ('sfc', date_range)})
atm_mo = atm_mo.set_index(sfc="time")

cpl_mo = xr.open_dataset("storm_center_lingling_case_CPL_MO_1-1-2"+tag+".nc")
cpl_mo = cpl_mo.assign_coords({"time": ('sfc', date_range)})
cpl_mo = cpl_mo.set_index(sfc="time")

cpl_moa = xr.open_dataset("storm_center_lingling_case_CPL_MO2_1-1-2"+tag+".nc")
cpl_moa = cpl_moa.assign_coords({"time": ('sfc', date_range)})
cpl_moa = cpl_moa.set_index(sfc="time")

#atm_mo = atm_mo.isel(sfc=slice(0,37))
#cpl_mo = cpl_mo.isel(sfc=slice(0,37))
#cpl_moa= cpl_moa.isel(sfc=slice(0,37))

windows = 3
atm_mo = atm_mo.rolling(sfc=windows, center=True).mean()
cpl_mo = cpl_mo.rolling(sfc=windows, center=True).mean()
cpl_moa = cpl_moa.rolling(sfc=windows, center=True).mean()

ds1, ds2, ds3 = atm_mo, cpl_mo, cpl_moa
ds1 = ds1.isel(sfc=slice(0,37), level=slice(0,17))
ds2 = ds2.isel(sfc=slice(0,37), level=slice(0,17))
ds3 = ds3.isel(sfc=slice(0,37), level=slice(0,17))
#ds1 = ds1.isel(level=slice(0,17))
#ds2 = ds2.isel(level=slice(0,17))
#ds3 = ds3.isel(level=slice(0,17))
print(ds1)

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(7, 4))#, sharex=True, sharey=True )

mesh = ax.contourf(ds1.sfc, ds1.level, ds3.eth.T-ds2.eth.T, 
        levels=np.arange(-1.5,1.7,0.2), cmap=bwr, extend='both')
        #levels=np.arange(-1.6,1.8,0.2), cmap=bwr, extend='both')

CS = ax.contour(ds1.sfc, ds1.level, -(ds3.omega.T-ds2.omega.T), 
        levels=np.arange(-0.7,0.9,0.2), 
        colors='k', linewidths=1)
CS.levels = [nf(val) for val in CS.levels]
ax.clabel(CS, CS.levels, fmt='%r', colors='k', inline=True, fontsize=9)

ax.set_yscale('symlog')
ax.set_yticklabels(np.arange(1000, 100, -100), fontsize=10)
ax.set_ylim(ds1.level.max(), ds1.level.min())
ax.set_yticks(np.arange(1000, 100, -100))
ax.set_ylabel('Pressure '+r'$[hPa]$', color='k', fontsize=12)

ax.set_title("CPL_down minus CPL_nodown", loc='center', fontsize=13)
ax.yaxis.grid(True, ls=':')
ax.tick_params(direction="in")
ax.set_xticks(pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H'))
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ\n%b %d'))

arrow = mpatches.FancyArrowPatch((0.6, 0), (0.6, 0.3), mutation_scale=40, transform=ax.transAxes, color='b', alpha=0.7)
ax.add_patch(arrow)

arrow = mpatches.FancyArrowPatch((0.75, 0), (0.75, 0.3), mutation_scale=40, transform=ax.transAxes, color='r', alpha=0.7)
ax.add_patch(arrow)

arrow = mpatches.FancyArrowPatch((0.875, 0.6), (0.875, 0.3), mutation_scale=40, transform=ax.transAxes, color='r', alpha=0.7)
ax.add_patch(arrow)

#ax.legend()
uw_arr = u'$\u2191$'
dw_arr = u'$\u2193$'
labels = [ 'weakening of the upward motion', 'strengthening of the upward motion',
        'weakening of the downward motion', 'strengthening of the downward motion']

elements = [ Line2D([0], [0], lw=0, marker=uw_arr, color='b', label=labels[0], mfc='b', mec='b', ms=10, alpha=0.7),
        Line2D([0], [0], lw=0, marker=uw_arr, color='r', label=labels[1], mfc='r', mec='r', ms=10, alpha=0.7),
        #Line2D([0], [0], lw=0, marker=dw_arr, color='b', label=labels[2], mfc='b', mec='b', ms=10, alpha=0.7),
        Line2D([0], [0], lw=0, marker=dw_arr, color='r', label=labels[3], mfc='r', mec='r', ms=10, alpha=0.7),
        ]
ax.legend(handles=elements, loc='lower left', framealpha=0.5, prop={'size': 9})

cbar = fig.colorbar(mesh, ax=ax, aspect=40, pad=0, orientation='horizontal')
#cbar.set_label(r'$\Delta \theta_{e}$ [shaded, K]'+' & '+r'$\Delta q$ [contours]', rotation=-90, va='bottom', fontsize=10, )
cbar.ax.set_title(r'$\Delta \theta _{e}$ [shaded, K]'+' & '+r'$\Delta omega$ [contours, Pa/s]', fontsize=12, )

plt.show()

fig.savefig('fig_05_TC_center_structure.png', dpi=300, bbox_inches='tight')
