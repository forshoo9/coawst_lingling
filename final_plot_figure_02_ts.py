import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mdates

from yang_func import *
from netCDF4 import Dataset

loc = plticker.MultipleLocator(base=0.5)

date_range = pd.date_range('2019-09-06 00:00:00', '2019-09-08 18:00:00', freq='H')

obs_lat, obs_lon = [ 32.12295277777780, 125.182447222222 ]
xi = ([obs_lon, obs_lat])

ix, iy, ik = [ 53, 83, 1 ] 

skipo = (slice(0,None), slice(ix,ix+1), slice(iy,iy+1))
skipf = (slice(0,None), slice(ix-ik,ix+ik+1), slice(iy-ik,iy+ik+1))

of = pd.read_csv("IORS_LINGLING.csv", index_col='date', parse_dates=True)
of.index = of.index.tz_localize('Asia/Seoul').tz_convert('UTC')
of = of.resample('60min').mean()
ods = xr.Dataset.from_dataframe(of) # I-ORS 
ds0 = xr.open_dataset("IORS_from_wrf3roms1_1-1-2.nc") # WRF-ROMS
ds1 = xr.open_dataset("IORS_from_wrf3roms1_adj_Qs_1-1-2.nc") # WRF-ROMS
ds_nocpl = xr.open_dataset("IORS_from_wrf3_1-1-2.nc") # WRF only
print(ds_nocpl)

flx_eddy = pd.read_csv("/home/data/WRF/flux_team/ieodo/flux_obs.csv", header=None, 
names=['sensible','latent','ustar'])
flx_eddy = flx_eddy.replace(-999., np.nan)
flx_bulk = xr.open_dataset("/home/data/WRF/flux_team/ieodo/lingling_hourly.nc")
print(flx_eddy)

#--------
# Plots
#--------
print("")
print("Plotting time series ...")
#fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(4, 8) )
fig = plt.figure(constrained_layout=True, figsize=(8, 4), )
gs = fig.add_gridspec(2, 2) 

ax = fig.add_subplot(gs[:,0])
print("--- SST ... ")

ax.plot(ds0.ocean_time, ds_nocpl.sst_5m.values, lw=2, color='b', label='NOCPL (SST)')
ax.plot(ds0.ocean_time, ds0.water_temp.sel(z_r=-5).values, lw=2, color='r', label='CPL_down (SST)')
ax.plot(ds0.ocean_time, ds0.water_temp.sel(z_r=-39).values, lw=2, 
        ls='--', color='r', label='CPL_down (T$_{38m}$)')
ax.plot(ds0.ocean_time, ods.SST1.values, lw=2, color='k', label='IORS (SST)')
ax.plot(ds0.ocean_time, ods.SST5.values, lw=2, 
        ls='--', color='k', label='IORS (T$_{38m}$)')

ax.axvline(x='2019-09-06 14:00:00',color='g', lw=2, zorder=0)
#ax.text('2019-09-07 09:00:00', 26.2, 'CPL', color='red', fontsize=12)
#ax.text('2019-09-07 09:00:00', 24.2, 'IORS', color='k', fontsize=12)

#ax.text('2019-09-07 14:00:00', 15.5, '(solid) SST')
#ax.text('2019-09-07 14:00:00', 15, '(dashed) T_38m')

ax.legend(prop={'size': 8}, loc='lower right')

ax.set_xlabel('Date '+r'$[HHZ/DD]$', fontsize=12)
#ax.set_ylabel(r'$[degC]$', fontsize=12)
ax.set_title('Temperature [degC]', loc='center', fontsize=13)
#ax.fill_between(ds0.Time.isel(Time=slice(3,27)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())
ax.set_xlim(['2019-09-06 00:00:00','2019-09-08 18:00:00'])
ax.set_ylim([14,28])
ax.grid(True, color='grey', ls='--', lw=0.5 )
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ/%d'))

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ/%d'))

ax.text(x=0.015, y=1.005, s='a', fontsize=15, weight='bold', ha='center',va='bottom', transform=ax.transAxes)


ax = fig.add_subplot(gs[0,1])
print("--- LHF ... ")

#ax.plot(ds0.Time, flx_bulk.lhf_c36, color='gray', lw=2, label='IORS_bulk')
ax.plot(ds0.Time, flx_eddy.latent, color='k', lw=1.5, ls='-', marker='.', ms=7, label='IORS')
ax.plot(ds0.Time, ds_nocpl.latent, color='b', lw=2, label='NOCPL')
ax.plot(ds0.Time, ds0.latent, color='red', lw=2, label='CPL_down')
#ax.fill_between(ds0.Time.isel(Time=slice(3,27)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())

ax.axhline(y=0, color='k', ls='-', lw=1)
ax.axvline(x='2019-09-06 14:00:00',color='g', lw=2, zorder=0)
ax.set_ylim([-100,350])
#ax.set_ylim([-100,260])
#ax.set_ylabel(r'$[W/m^{2}]$', fontsize=12)
ax.set_title(r'Latent heat flux $[W/m^{2}]$', loc='center', fontsize=13)
ax.grid(True, color='grey', ls='--', lw=0.5 )
ax.legend(prop={'size': 8}, ncol=3, columnspacing=0.4, loc='upper right')

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ/%d'))

ax.text(x=0.015, y=1.005, s='b', fontsize=15, weight='bold', ha='center',va='bottom', transform=ax.transAxes)


ax = fig.add_subplot(gs[1,1])
print("--- SHF ... ")

#ax.plot(ds0.Time, flx_bulk.shf_c36, color='gray', lw=2, label='IORS_bulk')
ax.plot(ds0.Time, flx_eddy.sensible, color='k', lw=1.5, ls='-', marker='.', ms=7, label='IORS')
ax.plot(ds0.Time, ds_nocpl.sensible, color='b', lw=2, label='NOCPL')
ax.plot(ds0.Time, ds0.sensible, color='red', lw=2, label='CPL_down')
#ax.fill_between(ds0.Time.isel(Time=slice(3,27)), 0, 1, 
#        color='gold', alpha=0.4, transform=ax.get_xaxis_transform())

ax.axhline(y=0, color='k', ls='-', lw=1)
ax.axvline(x='2019-09-06 14:00:00',color='g', lw=2, zorder=0)
ax.set_ylim([-70,50])
#ax.set_ylabel(r'$[W/m^{2}]$', fontsize=12)
ax.set_title(r'Sensible heat flux $[W/m^{2}]$', loc='center', fontsize=13)
ax.grid(True, color='grey', ls='--', lw=0.5 )
ax.legend(prop={'size': 8}, ncol=3, columnspacing=0.4, loc='lower right')

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%HZ/%d'))

ax.text(x=0.015, y=1.005, s='c', fontsize=15, weight='bold', ha='center',va='bottom', transform=ax.transAxes)

ax.set_xlabel('Date [HHZ/DD]', fontsize=12)

plt.show()

fig.savefig('fig_02_IORS_sst_ts.png', dpi=500, bbox_inches='tight')
