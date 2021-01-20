import proplot as plot

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset, num2date

import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd

import scipy.ndimage as ndimage
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.interpolate import log_interpolate_1d, cross_section
from metpy.plots import add_metpy_logo, add_timestamp
from metpy.units import units

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, ALL_TIMES,
        cartopy_ylim, latlon_coords, interplevel, vinterp, extract_vars,
                 omp_enabled, omp_get_num_procs, omp_set_num_threads,
                 extract_times, ll_to_xy, get_basemap)

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def metpy_read_wrf_cross(fname, plevels, tidx_in, lons_out, lats_out, start, end):

    ds = xr.open_dataset(fname).metpy.parse_cf().squeeze()
    ds = ds.isel(Time=tidx)
    print(ds)


    ds1 = xr.Dataset()

    p = units.Quantity(to_np(ds.p), 'hPa')
    z = units.Quantity(to_np(ds.z), 'meter')
    u = units.Quantity(to_np(ds.u), 'm/s')
    v = units.Quantity(to_np(ds.v), 'm/s')
    w = units.Quantity(to_np(ds.w), 'm/s')
    tk = units.Quantity(to_np(ds.tk), 'kelvin')
    th = units.Quantity(to_np(ds.th), 'kelvin')
    eth = units.Quantity(to_np(ds.eth), 'kelvin')
    wspd = units.Quantity(to_np(ds.wspd), 'm/s')
    omega = units.Quantity(to_np(ds.omega), 'Pa/s')

    plevels_unit = plevels * units.hPa
    z,u,v,w,tk,th,eth,wspd,omega = log_interpolate_1d(plevels_unit,p,z,u,v,w,tk,th,eth,wspd,omega,axis=0)
    
    coords, dims = [plevs,ds.lat.values,ds.lon.values], ["level","lat","lon"]
    for name, var in zip(['z','u','v','w','tk','th','eth','wspd','omega'], [z,u,v,w,tk,th,eth,wspd,omega]):
        #g = ndimage.gaussian_filter(var, sigma=3, order=0)
        ds1[name] = xr.DataArray(to_np(mpcalc.smooth_n_point(var, 9)), coords=coords, dims=dims)

    dx, dy = mpcalc.lat_lon_grid_deltas(ds.lon.values * units('degrees_E'), 
            ds.lat.values * units('degrees_N'))

    # Calculate temperature advection using metpy function
    for i, plev in enumerate(plevs):

        adv = mpcalc.advection(eth[i,:,:], [u[i,:,:], v[i,:,:]], (dx, dy), dim_order='yx') * units('K/sec')
        adv = ndimage.gaussian_filter(adv, sigma=3, order=0) * units('K/sec')
        ds1['eth_adv_{:03d}'.format(plev)] = xr.DataArray(np.array(adv), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

        div = mpcalc.divergence(u[i,:,:], v[i,:,:], dx, dy, dim_order='yx')
        div = ndimage.gaussian_filter(div, sigma=3, order=0) * units('1/sec')
        ds1['div_{:03d}'.format(plev)] = xr.DataArray(np.array(div), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

    ds1['accrain'] = xr.DataArray(ds.accrain.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    eth2 = mpcalc.equivalent_potential_temperature(ds.slp.values*units.hPa, ds.t2m.values*units('K'), ds.td2.values*units('celsius'))
    ds1['eth2'] = xr.DataArray(eth2, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    #ds1['sst'] = xr.DataArray(ndimage.gaussian_filter(ds.sst.values, sigma=3, order=0)-273.15, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1 = ds1.metpy.parse_cf().squeeze()

    cross = cross_section(ds1, start, end).set_coords(('lat','lon'))
    cross.u.attrs['units'] = 'm/s'
    cross.v.attrs['units'] = 'm/s'

    cross['t_wind'], cross['n_wind'] = mpcalc.cross_section_components(
            cross['u'], cross['v'])

    weights = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    weighted = ds_weighted.mean(("lat"))

    return ds1, cross, weighted


# draw filled contours.
clevs = [0, 2, 5, 10, 15, 20, 30, 40, 50, 70, 100] #, 150, 200, 250, 300, 400, 500, 600, 750]
cmap_data = [(1.0, 1.0, 1.0),
             #(0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
             (0.0, 1.0, 1.0),
             #(0.0, 0.8784313797950745, 0.501960813999176),
             (0.0, 0.7529411911964417, 0.0),
             (0.501960813999176, 0.8784313797950745, 0.0),
             (1.0, 1.0, 0.0),
             (1.0, 0.6274510025978088, 0.0),
             (1.0, 0.0, 0.0),
             (1.0, 0.125490203499794, 0.501960813999176),
             (0.9411764740943909, 0.250980406999588, 1.0),
             (0.501960813999176, 0.125490203499794, 1.0),
             (0.250980406999588, 0.250980406999588, 1.0),]
cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
norm = mcolors.BoundaryNorm(clevs, cmap.N)
# In future MetPy
# norm, cmap = ctables.registry.get_with_boundaries('precipitation', clevs)

low = cm.GnBu_r(np.linspace(0, 0.75, 128))
mid = np.ones((30,4))
high = cm.YlOrRd(np.linspace(0.0, 0.95, 128))
colors = np.vstack((low, mid, high))
bwr = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)
bwr.set_under = 'darkbrown'
bwr.set_over = 'magenta'
#bwr.set_bad = 'k'

low = cm.YlOrBr_r(np.linspace(0, 0.9, 128))
mid = np.ones((30,4))
high = cm.BuGn(np.linspace(0, 0.95, 128))
colors = np.vstack((low, mid, high))
drywet = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)

#fwrf1 = "./outputs/wrf3roms1/1-1-2/wrfout_d01_2019-09-06_00:00:00"
#fwrf2 = "./outputs/wrf3roms1_adj_Qs/1-1-2/wrfout_d01_2019-09-06_00:00:00"

minLon, maxLon, minLat, maxLat = [119,130,27,40]
hres = 0.1
lons = np.arange(minLon,maxLon+hres,hres) # analysis domain c    overing [118.5, 128, 26.5, 42.]
lats = np.arange(minLat,maxLat+hres,hres)

plevs = np.arange(950,250,-50) #[925, 900, 850, 700, 500, 300]
print(plevs)

#start = (33,123)
#end = (28,129)
#start = (33.5,122)
start = (35,123)
end = (31,129)
#start = (35.5,123)
#end = (30.5,129)
obs_lat, obs_lon = [ 32.12295277777780, 125.182447222222 ]

tidx = 60
data1, cross1, wgt1 = metpy_read_wrf_cross('data/metpy_cpl_nodown.nc', plevs, tidx, lons, lats, start, end)
data2, cross2, wgt2 = metpy_read_wrf_cross('data/metpy_cpl_down.nc', plevs, tidx, lons, lats, start, end)
print(cross2)

#-----------------------------------
# Define the figure object and primary axes
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 4), sharex=True)
#fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 8), sharex=True)

#-----------------------------------
data = data2-data1
cross = cross2-cross1
wgt = wgt2-wgt1

#-----------------------------------
#ax = axes[0]
#
title = r"Cross-section from (35$\degree$N, 123$\degree$E) to (31$\degree$N, 129$\degree$E) on September 08 at 1200Z"
ax.text(x=0.5, y=1.1, s=title, fontsize=13, ha='center',va='bottom', transform=ax.transAxes)

#ax.set_title(r'$\Delta$PRECIP along the cross-line', fontsize=12)
#ax.plot(cross.lon, cross.accrain, color='k')
#ax.fill_between(cross.lon, cross.accrain, 0, where=(cross.accrain.values>=0), color='forestgreen',interpolate=True, label='wet')
#ax.fill_between(cross.lon, cross.accrain, 0, where=(cross.accrain.values< 0), color='darkorange', interpolate=True, label='dry')
#ax.legend(loc='lower left')
##ax.legend(loc='upper right')
#ax.set_xlim(start[1],end[1])
#ax.set_ylabel('[mm]', fontsize=11)
#ax.text(x=0.01, y=1.005, s='a', fontsize=16, ha='center',va='bottom', weight='bold', transform=ax.transAxes)
#ax.tick_params(labelsize=9.5) 

#-----------------------------------
#ax = axes[1]
#
ax.set_title(r'Differences of $\theta_{e}$ [shaded], $-\omega$ [contours], and wind vectors between both experiments', fontsize=11)

#cf_levs = np.arange(-3,3.5,0.5)
#cf_levs = np.arange(-2.7,2.7,0.6)
cs_levs = np.arange(-3.5,4.5,1)

import matplotlib as mpl
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

#cf = ax.contourf(cross['lon'], cross['level'], cross['eth'], 
#        levels=cf_levs, cmap=bwr, extend='both')

shifted_cmap = shiftedColorMap(bwr, midpoint=0.7, name='shifted') 
#norm = MidpointNormalize(vmin=-6, vmax=3, midpoint=0)
cf = ax.contourf(cross['lon'], cross['level'], cross['eth'], 
        levels=np.arange(-4,2.5,0.5), cmap=shifted_cmap, extend='both')
cbar = fig.colorbar(cf, ax=ax, shrink=0.8, aspect=60, pad=-0.015, ticks=np.arange(-4,2.5,0.5))
#, ticks=np.arange(-10,12,2)/10)
cbar.ax.text(x=-1, y=1.025, s='warmer air', fontsize=10, transform=cbar.ax.transAxes)
cbar.ax.text(x=-1, y=-0.065, s='colder air', fontsize=10, transform=cbar.ax.transAxes)
cbar.set_label('[K]', fontsize=11, rotation=-90, va='bottom')
cbar.ax.tick_params(labelsize=9.5) 
cbar.ax.tick_params(which='minor', length=0)

cs = ax.contour(cross['lon'], cross['level'], -cross['omega'], 
        levels=cs_levs[np.where(cs_levs!=0)], colors='k', linewidths=1.5)
ax.clabel(cs, fontsize=8, inline=1, inline_spacing=1, fmt='%0.1f')
    
skip = slice(None,None, 5)
q = ax.quiver(cross['lon'][skip], cross['level'][:], 
        cross['t_wind'][:,skip], cross['n_wind'][:,skip],
        headwidth=5, width=0.0035, scale=100, zorder=11, color='grey') 
ax.quiverkey(q, X=0.975, Y=1.08, U=3, label='3 m/s', labelpos='S', zorder=20)

# Adjust the y-axis to be logarithmic
ax.set_yscale('symlog')
ax.set_yticklabels(np.arange(1000, 200, -100))
ax.set_ylim(cross['level'].max(), cross['level'].min())
ax.set_yticks(np.arange(1000, 200, -100))
ax.set_ylabel('Pressure [hPa]', fontsize=11)

ax.set_xticks(np.arange(123,130,1))
ax.set_xticklabels([ r"{:d}$\degree$E".format(i) for i in np.arange(123,130,1)])
ax.set_xlabel('Longitude along the cross-line (inset)', fontsize=11)
ax.tick_params(labelsize=9.5) 
    
#-----------------------------------
# Define the CRS and inset axes
data_crs = data2['accrain'].metpy.cartopy_crs
ax_inset = fig.add_axes([0.65, 0.495, 0.3, 0.3], projection=data_crs)
gl = ax_inset.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=':')
gl.xlabels_top, gl.xlabels_bottom, gl.ylabels_left, gl.ylabels_right = [False, True, True, False]
gl.xformatter, gl.yformatter = [LONGITUDE_FORMATTER, LATITUDE_FORMATTER]
ax_inset.tick_params(labelsize=8) 
cf = ax_inset.contourf(data['lon'], data['lat'], mpcalc.smooth_n_point(data['accrain'], 9), 
        levels=np.arange(-22,26,4), cmap=drywet, extend='both')
        #levels=np.arange(-20,24,4), cmap=drywet, extend='both')
axins = inset_axes(ax_inset, width="5%", height="70%", loc='lower left')
cbar = fig.colorbar(cf, cax=axins, orientation="vertical", ticks=[-18,-6,6,18])
cbar.ax.tick_params(which='both', direction='in')
cbar.ax.tick_params(which='both', length=0)

#ax_inset.plot(obs_lon, obs_lat, color='k', ms=10, marker='*', zorder=100, alpha=0.5 )
ax_inset.scatter([start[1],end[1]], [start[0],end[0]], c='r', zorder=2)
ax_inset.plot(cross['lon'], cross['lat'], c='r', zorder=2)

ax_inset.text(0.5, 1.005, s=r'$\Delta$PRECIP [mm]', color='k', fontsize=11, ha='center', va='bottom',
        transform=ax_inset.transAxes)
ax_inset.text(0.55, 0.35, s='cross-line', color='r', fontsize=11, ha='center', va='center',
        transform=ax_inset.transAxes, rotation=-34)

ax_inset.coastlines(color='grey')
#ax_inset.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k', alpha=0.2, zorder=0)
ax_inset.set_extent([start[1]-1, end[1]+1, end[0]-1, start[0]+1])
#-----------------------------------

plt.show()

fig.savefig('fig_07_cross_section.png'.format(tidx), dpi=500, bbox_inches='tight')

