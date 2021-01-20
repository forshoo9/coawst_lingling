import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
from cartopy.feature import OCEAN,LAND,LAKES
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
 
from yang_func import *


low = cm.GnBu_r(np.linspace(0.01,0.9, 128))
mid = np.ones((50,4))
high = cm.YlOrRd(np.linspace(0.1, 0.95, 128))
colors = np.vstack((low, mid, high))
bwr = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)
bwr.set_under = 'darkblue'
bwr.set_over = 'darkbrown'
bwr.set_bad = 'k'

rainbow = cm.get_cmap("jet")
rainbow.set_under("magenta")
rainbow.set_over("darkred")


#loc = plticker.MultipleLocator(base=1/2)

date_range = pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H')

fobs = 'jtwc_lingling.csv'
df = pd.read_csv(fobs, index_col='date', parse_dates=True).dropna(axis=1)
df = df.drop(columns=['index'])
df = df.loc[date_range[0]:date_range[-1],:]

tf = pd.read_csv('storm_center_lingling_case_total_adj_Qs_100km.csv', index_col='date', parse_dates=True)

df1 = tf[tf.cases=='ATM_MO']
df2 = tf[tf.cases=='CPL_MO']
df3 = tf[tf.cases=='CPL_MO2']

obs_lat, obs_lon = [ 32.12295277777780, 125.182447222222 ]
#obs_lat, obs_lon = [ 32.123, 125.183 ]
Gobs_lat, Gobs_lon = [ 33.942, 124.593 ]


infile_d01 = '../outputs/wrf3roms1_adj_Qs/1-1-2/wrfout_d01_2019-09-06_00:00:00'
cart_proj, xlim_d01, ylim_d01, lons, lats = get_plot_element(infile_d01)

froms = '../outputs/wrf3roms1_adj_Qs/1-1-2/Lingling_ocean_his_standard_zlevs_under_100m.nc'
romsin = xr.open_dataset(froms)
romsin = romsin.rename({"lat_rho": "lat", "lon_rho": "lon"})

sst0 = romsin.isel(ocean_time=0, z_r=0).temp
sst1 = romsin.isel(ocean_time=48, z_r=0).temp
delta_sst = sst1-sst0


ds0 = xr.open_dataset("IORS_from_wrf3roms1_1-1-2_meridional_section_under_100m.nc")
ds = ds0.isel(lon=0)
print(ds)


fig = plt.figure(constrained_layout=True, figsize=(9, 7) )
gs = fig.add_gridspec(1, 2, width_ratios=(2,1)) 


print("")
print("Plotting time series ...")

proj = ccrs.PlateCarree()
ax = fig.add_subplot(gs[0], projection=proj) # projection=cart_proj
 
ax.add_feature(OCEAN, lw=0.5, ec='k', zorder=0, facecolor='w')
ax.add_feature(LAND,  lw=0.5, ec='k', zorder=1, facecolor='cornsilk')
ax.add_feature(LAKES, lw=0.5, ec='k', zorder=2, facecolor='w')
ax.coastlines('50m', linewidth=0.5, )
ax.set_extent([118.5, 127.5, 27.5, 41.5], crs=proj)
ax.set_title(r"TC Tracks", fontsize=13)

ax.plot(df.lon.values, df.lat.values, color='k', label='JTWC', lw=1.5, marker='s', mfc='none', ms=8, transform=proj)


ds1 = ds0.isel(lat=slice(5,51),lon=slice(5,51))
ax.plot(ds1.lon.values, ds1.lat.values, color='g', lw=2)

df1 = df1[::6]
df2 = df2[::6]
df3 = df3[::6]
ax.plot(df1.lon.values, df1.lat.values, color='b', lw=1.5, transform=proj)
ax.plot(df3.lon.values, df3.lat.values, color='r', lw=1.5, transform=proj)
#
ax.text(125.2, 35.5, 'cross-line', color='g', fontsize=12, ha='left', va='center', rotation=-90)
#
date_range = pd.date_range('2019-09-06 00:00:00', '2019-09-07 12:00:00', freq='6H')

for i, date in enumerate(date_range):

    lon, lat = [df1.loc[date,'lon'], df1.loc[date,'lat']]
    ax.plot(lon, lat, color='b', ms=8, marker='o', mfc='none', transform=proj)

    lon, lat = [df3.loc[date,'lon'], df3.loc[date,'lat']]
    ax.plot(lon, lat, color='r', ms=7, marker='x', mfc='none', transform=proj)

    at_x, at_y = ax.projection.transform_point(lon, lat, src_crs=proj)
    ax.annotate(date.strftime("%HZ/%d"), size=11, xy=(at_x, at_y), xytext=(-27.5,0),
            textcoords='offset points', va="center", ha="center", 
            #bbox=dict(boxstyle="round", fc="w", alpha=0.75),
            #arrowprops=dict(arrowstyle='->', color='k', ls='--', lw=1, ), 
            transform=proj,)

lon, lat = [df1.loc[date,'lon'], df1.loc[date,'lat']]
ax.plot(lon, lat, color='b', ms=8, marker='o', mfc='none', label='NOCPL', transform=proj)

lon, lat = [df3.loc[date,'lon'], df3.loc[date,'lat']]
ax.plot(lon, lat, color='r', ms=7, marker='x', mfc='none', label='CPL_down', transform=proj)

ax.plot(obs_lon, obs_lat, color='k', ms=8, marker='*', transform=proj )

ax.legend(loc='lower left', prop={'size': 9}, frameon=True)
#
##ax.set_extent([118.5, 129, 27, 41], crs=proj)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlabel('Longitude', fontsize=12)

gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=':',
        xlocs=[118,120,122,124,126,128,130], auto_inline=False,
        xformatter=LONGITUDE_FORMATTER, yformatter=LATITUDE_FORMATTER)
gl.top_labels = False
gl.right_labels = False

ax.text(x=-0.125, y=0.5, s='Latitude', fontsize=11, ha='center',va='center', rotation=90, transform=ax.transAxes)
ax.text(x=0.5, y=-0.05, s='Longitude', fontsize=11, ha='center',va='center', transform=ax.transAxes)
ax.text(x=0.025, y=1.005, s='a', fontsize=15, weight='bold', ha='center',va='bottom', transform=ax.transAxes)

ax = fig.add_subplot(gs[1])

mesh = ax.contourf(-1*ds.z_r, ds.lat, ds.water_temp.T.isel(ocean_time=0), extend='both',
         levels=np.arange(10,32,2), cmap=rainbow, zorder=1)

axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.85, 0.55, 0.05, 0.4), 
        bbox_transform=ax.transAxes, loc=3, borderpad=0)
cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")#, ticks=[-18,-6,6,18])
cbar.ax.tick_params(which='both', direction='in')
cbar.ax.tick_params(which='both', length=0)
cbar.ax.set_title('[degC]', fontsize=11)
#cbar = fig.colorbar(mesh, ax=ax, aspect=95)#, orientation='horizontal')

#CS = ax.contour(-1*ds.z_r, ds.lat, ds.water_temp.T.isel(ocean_time=0),
#        levels=np.arange(10,38,4), colors='k', linewidths=1, zorder=1)
#CS.levels = [nf(val) for val in CS.levels]
#ax.clabel(CS, CS.levels, fmt='%r', colors='k', inline=True, fontsize=10)

CS = ax.plot(ds.oml_depth.isel(ocean_time=0), ds.lat, color='w', lw=4)

ax.axhline(y=32.123, color='k', ls='--', lw=1, zorder=10)
#ax.scatter(107, 32.123, s=100, c='k', marker='*', zorder=20)
#ax.text(105, 32.123, 'IORS', color='k', fontsize=11, ha='right', va='center', zorder=13,
#        bbox=dict(boxstyle="square", ec='none', fc='white'),)

ax.text(2, 30.7, 'OML', fontsize=12, color='w', weight='bold')

ax.set_xlim([0,110])
ax.set_ylim([28,37])
ax.set_xlabel('Depth', fontsize=12)
ax.set_ylabel('Latitude along the cross-line', fontsize=12)
ax.set_title('Initial T_prof', loc='center', fontsize=13)
#ax.set_title('    Initial T_prof along the cross-line', loc='left', fontsize=12.5)
#ax.spines['left'].set_color('red')
#ax.spines['left'].set_linewidth(3)
ax.set_xticks(np.arange(10,120,20))
ax.set_xticklabels([ "{:d}m".format(i) for i in np.arange(10,120,20)])
ax.set_yticks(np.arange(28,38,1))
ax.set_yticklabels([ r"{:d}$\degree$N".format(i) for i in np.arange(28,38,1)])
ax.grid(True, color='grey', ls='--', lw=0.5, zorder=2 )
ax.tick_params(direction="in")
plt.setp(ax.get_yticklabels(), fontsize=10)
plt.setp(ax.get_xticklabels(), fontsize=10)
ax.text(x=0.025, y=1.005, s='b', fontsize=15, weight='bold', ha='center',va='bottom', transform=ax.transAxes)

plt.show()
fig.savefig('fig_03_TC_tracks.png', dpi=500, bbox_inches='tight')
