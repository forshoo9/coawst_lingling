import numpy as np
import xarray as xr
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 
import cartopy.crs as ccrs
from cartopy.feature import OCEAN,LAND,LAKES
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from wrf import (getvar, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim, to_np)

from yang_func import *


obs_lat, obs_lon = [ 32.12295277777780, 125.182447222222 ]

roms = xr.open_dataset('../outputs/wrf3roms1/Lingling_ocean_avg.nc')

out_dir = '../outputs/wrf3roms1/'
infile_d01 = out_dir+'/wrfout_d01_2019-09-06_00:00:00'
cart_proj, xlim_d01, ylim_d01, lons, lats = get_plot_element(infile_d01)
 
infile_d02 = out_dir+'/wrfout_d02_2019-09-06_00:00:00'
_, xlim_d02, ylim_d02, _, _ = get_plot_element(infile_d02)
 
infile_d03 = out_dir+'/wrfout_d03_2019-09-06_00:00:00'
_, xlim_d03, ylim_d03, _, _ = get_plot_element(infile_d03)
 
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=cart_proj)
proj = ccrs.PlateCarree()
 
# d01
xlim, ylim = [xlim_d01, ylim_d01]
ax.set_xlim([xlim[0]-(xlim[1]-xlim[0])/15, xlim[1]+(xlim[1]-xlim[0])/15])
ax.set_ylim([ylim[0]-(ylim[1]-ylim[0])/15, ylim[1]+(ylim[1]-ylim[0])/15])
 
# d01 box
for i, (xlim, ylim, s,c) in enumerate(
        zip([xlim_d01, xlim_d02, xlim_d03], [ylim_d01, ylim_d02, ylim_d03],
            [15, 12, 9], ['b','g','r'])):

    ax.add_patch(mpl.patches.Rectangle((xlim[0], ylim[0]),
        xlim[1]-xlim[0], ylim[1]-ylim[0],
        fill=None, lw=1.5, edgecolor='r', zorder=10))

ax.add_feature(OCEAN, lw=0.5, ec='k', zorder=0, facecolor='w')
ax.add_feature(LAND,  lw=0.5, ec='k', zorder=1, facecolor='cornsilk')
ax.add_feature(LAKES, lw=0.5, ec='k', zorder=2, facecolor='w')
#ax.coastlines('50m', linewidth=1.5, )

white = np.ones((10,4))
high = cm.YlGnBu(np.linspace(0, 1, 128))
colors = np.vstack((white, high))
my_color = LinearSegmentedColormap.from_list('my_colormap',colors, N=24)
#

CS = ax.contour(to_np(lons), to_np(lats), roms.h, 
        levels=[10,50,100,200,500], linewidths=1.25, colors='k',  transform=proj)
CS.levels = [nf(val) for val in CS.levels]
ax.clabel(CS, CS.levels, fmt='%r', inline=True, fontsize=9, )

cn = ax.contourf(to_np(lons), to_np(lats), roms.h, np.arange(0,7000,500), cmap=my_color, transform=proj, )
cbar = fig.colorbar(cn, ax=ax, shrink=0.6, )
cbar.set_label("ROMS Bathymetry (m)", rotation=-90, va='bottom', fontsize=14)
cbar.add_lines(CS)

lonmin, lonmax, latmin, latmax = [118, 128, 27, 41]
#lonmin, lonmax, latmin, latmax = [118.5, 128, 26.5, 42.]
xs = [lonmin, lonmax, lonmax, lonmin, lonmin]
ys = [latmin, latmin, latmax, latmax, latmin]
ax.plot(xs, ys, transform=proj, linewidth=2., color='g', ls='--')
ax.text(x=0.42,y=0.815, s='Analysis region', rotation=-2, fontsize=14, weight='bold', color='g', 
        ha='center', va='bottom', transform=ax.transAxes, )
ax.text(x=0.06,y=0.94, s='D1', fontsize=15, weight='bold', color='r', ha='left', va='bottom', transform=ax.transAxes, )
ax.text(x=0.395,y=0.655, s='D2', fontsize=13, weight='bold', color='r', ha='left', va='bottom', transform=ax.transAxes, )
ax.text(x=0.445,y=0.38, s='D3', fontsize=11, weight='bold', color='r', ha='left', va='bottom', transform=ax.transAxes, )
ax.text(x=0.39,y=0.5, s='Yellow Sea', fontsize=12, color='k', ha='center', va='bottom', transform=ax.transAxes, )
ax.text(x=0.39,y=0.375, s='YECS', fontsize=12, color='k', ha='center', va='bottom', transform=ax.transAxes, )
ax.text(x=0.475,y=0.24, s='East China Sea', fontsize=12, color='k', ha='center', va='bottom', transform=ax.transAxes, )
ax.text(x=0.8,y=0.15, s='North Pacific', fontsize=12, color='k', ha='center', va='bottom', transform=ax.transAxes, )
ax.text(x=0.8,y=0.75, s='East Sea', fontsize=12, color='k', ha='center', va='bottom', transform=ax.transAxes, )

ax.plot(obs_lon, obs_lat, color='b', ms=12, marker='*', transform=proj, )

fig.canvas.draw()
    
xticks=np.arange(112,140,2).tolist()
yticks=np.arange(12,50,2).tolist()

ax.gridlines(xlocs=xticks, ylocs=yticks, lw=1, color='gray', alpha=0.5, linestyle='--')

ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
lambert_xticks(ax, xticks)
lambert_yticks(ax, yticks)

#ax.set_title('WRF nested domain setup', size=18)

plt.show()

fig.savefig('fig_01_WRF_domain.png', dpi=500, bbox_inches='tight')
