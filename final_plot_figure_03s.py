import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc

import xarray as xr
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from yang_func import *

import matplotlib.ticker as plticker
import matplotlib.dates as mdates

loc = plticker.MultipleLocator(base=0.5)

date_range = pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H')

fobs = 'jtwc_lingling.csv'
of = pd.read_csv(fobs, index_col='date', parse_dates=True).dropna(axis=1)
of = of.drop(columns=['index'])
of = of.loc[date_range[0]:date_range[-1],:]

low = cm.GnBu_r(np.linspace(0.01,0.9, 128))
mid = np.ones((80-2*30,4))
high = cm.YlOrRd(np.linspace(0.1, 0.95, 128))
colors = np.vstack((low, mid, high))
bwr = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)
bwr.set_over = 'darkbrown'
bwr.set_bad = 'k'

#tf = pd.read_csv('storm_center_lingling_case_total_adj_Qs_200km.csv', index_col='date', parse_dates=True)
tf = pd.read_csv('storm_center_lingling_case_total_adj_Qs.csv', index_col='date', parse_dates=True)

resample = '3H'
windows = 1

atm_mo = tf[tf.cases=='ATM_MO']
cpl_mo = tf[tf.cases=='CPL_MO']
cpl_moa = tf[tf.cases=='CPL_MO2']

df1, df2, df3 = atm_mo, cpl_mo, cpl_moa

tag = '_50km'

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

windows = 3
atm_mo = atm_mo.rolling(sfc=windows, center=True).mean()
cpl_mo = cpl_mo.rolling(sfc=windows, center=True).mean()
cpl_moa = cpl_moa.rolling(sfc=windows, center=True).mean()

ds1, ds2, ds3 = atm_mo, cpl_mo, cpl_moa
ds1 = ds1.isel(sfc=slice(0,37), level=slice(0,17))
ds2 = ds2.isel(sfc=slice(0,37), level=slice(0,17))
ds3 = ds3.isel(sfc=slice(0,37), level=slice(0,17))

lw=2

loc = plticker.MultipleLocator(base=0.25)
fig, axes = plt.subplots(3,1, constrained_layout=True, figsize=(5, 6), sharex=True )

ax = axes[0] 
#ax.plot(ds1.sfc, ds1.sst, color='k', lw=lw, label='ATM')
ax.plot(ds1.sfc, ds2.sst, color='royalblue', lw=lw, label='CPL_nodown')
ax.plot(ds1.sfc, ds3.sst, color='r', lw=lw, label='CPL_down')

ax.legend(loc='best', prop={'size': 9}, frameon=False)
ax.set_title(r'SST', color='k', fontsize=12)
ax.set_ylabel(r'$[degC]$', color='k', fontsize=10)

ax.grid(True, ls=':')
ax.tick_params(direction="in")
ax.set_xticks(pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H'))
#ax.fill_between(ds1.sfc.isel(sfc=slice(18,22)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
#ax.fill_between(ds1.sfc.isel(sfc=slice(25,33)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
ax.set_ylim([20,30])
ax.set_xlim(['2019-09-06 00:00:00', '2019-09-07 12:00:00'])
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ\n%b %d'))

ax.text(x=0.0275, y=1.0075, s='a', fontsize=15, weight='bold', 
    ha='center',va='bottom', transform=ax.transAxes)

ax = axes[1] 
ax.legend(loc='best', prop={'size': 10}, frameon=False)
ax.set_title('Latent heat flux', color='k', fontsize=12)
#ax.set_title(r'Q$_{L}$', color='k', fontsize=12)
ax.set_ylabel(r'$[W/m^{2}]$', color='k', fontsize=10)

ax.plot(ds1.sfc, ds2.latent, color='royalblue', lw=lw, label='CPL_nodown')
ax.plot(ds1.sfc, ds3.latent, color='r', lw=lw, label='CPL_down')

ax.legend(loc='best', prop={'size': 9}, frameon=False)
ax.grid(True, ls=':')
ax.tick_params(direction="in")
ax.set_xticks(pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H'))
#ax.fill_between(ds1.sfc.isel(sfc=slice(18,22)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
#ax.fill_between(ds1.sfc.isel(sfc=slice(25,33)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())

ax.set_xlim(['2019-09-06 00:00:00', '2019-09-07 12:00:00'])
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ\n%b %d'))

ax.text(x=0.0275, y=1.0075, s='b', fontsize=15, weight='bold', 
    ha='center',va='bottom', transform=ax.transAxes)

ax = axes[2] 
ax.legend(loc='best', prop={'size': 10}, frameon=False)
#ax.set_title(r'Q$_{S}$', color='k', fontsize=12)
ax.set_title('Sensible heat flux', color='k', fontsize=12)
ax.set_ylabel(r'$[W/m^{2}]$', color='k', fontsize=10)

ax.plot(ds1.sfc, ds2.sensible, color='royalblue', lw=lw, label='CPL_nodown')
ax.plot(ds1.sfc, ds3.sensible, color='r', lw=lw, label='CPL_down')

ax.legend(loc='best', prop={'size': 9}, frameon=False)
ax.grid(True, ls=':')
ax.tick_params(direction="in")
ax.set_xticks(pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H'))
#ax.fill_between(ds1.sfc.isel(sfc=slice(18,22)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
#ax.fill_between(ds1.sfc.isel(sfc=slice(25,33)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
ax.set_xlim(['2019-09-06 00:00:00', '2019-09-07 12:00:00'])
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ\n%b %d'))

ax.text(x=0.0275, y=1.0075, s='c', fontsize=15, weight='bold', 
    ha='center',va='bottom', transform=ax.transAxes)

fig.savefig('final_fig_03_ts_TC_center_2.png', dpi=300, bbox_inches='tight')
plt.show()
