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
from metpy.interpolate import log_interpolate_1d
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

import matplotlib.patches as patches

def metpy_read_wrf(fwrf, fout, lons_out, lats_out):

    wrfin = Dataset(fwrf)
    ds = xr.Dataset()

    slp = getvar(wrfin, "slp", timeidx=ALL_TIMES)
    ds['slp'] = smooth2d(slp, 3, cenweight=4)

    lats, lons = latlon_coords(slp)
    ds['lats'] = lats
    ds['lons'] = lons

    rainc = []
    for tidx in range(len(ds.Time)):
        rainc0 = extract_vars(wrfin, timeidx=tidx, varnames=["RAINNC"] ).get("RAINNC")
        if tidx>0 : 
            rainc0 -= extract_vars(wrfin, timeidx=tidx-1, varnames=["RAINNC"] ).get("RAINNC")
        rainc.append(smooth2d(rainc0, 3, cenweight=4))
    ds['rain'] = xr.DataArray(rainc, ds.slp.coords, ds.slp.dims, ds.slp.attrs)
    ds['latent'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["LH"] ).get("LH")
    ds['u10'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["U10"] ).get("U10")
    ds['v10'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["V10"] ).get("V10")
    ds['t2m'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["T2"] ).get("T2")
    ds['q2m'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["Q2"] ).get("Q2")
    ds['sst'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["SST"] ).get("SST")
    ds['td2'] = getvar(wrfin, "td2", timeidx=ALL_TIMES)
    ds['th2'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["TH2"] ).get("TH2")
    ds['mask'] = extract_vars(wrfin, timeidx=ALL_TIMES, varnames=["LANDMASK"] ).get("LANDMASK")

    ds['p'] = getvar(wrfin, "pressure", timeidx=ALL_TIMES)
    ds['z'] = getvar(wrfin, "geopt", timeidx=ALL_TIMES)
    ds['u'] = getvar(wrfin, "ua", timeidx=ALL_TIMES)
    ds['v'] = getvar(wrfin, "va", timeidx=ALL_TIMES)
    ds['w'] = getvar(wrfin, "wa", timeidx=ALL_TIMES)
    ds['omega'] = getvar(wrfin, "omega", timeidx=ALL_TIMES)
    ds['tk'] = getvar(wrfin, "tk", timeidx=ALL_TIMES)
    ds['th'] = getvar(wrfin, "th", timeidx=ALL_TIMES, units='K')
    ds['eth'] = getvar(wrfin, "eth", timeidx=ALL_TIMES, units='K')
    ds['avo'] = getvar(wrfin, "avo", timeidx=ALL_TIMES)
    ds['pvo'] = getvar(wrfin, "pvo", timeidx=ALL_TIMES)
    ds['wspd'] = getvar(wrfin, "wspd_wdir", timeidx=ALL_TIMES, units="m/s")[0,:]

    ds = ds.rename({'south_north': 'eta_rho', 'west_east': 'xi_rho'})
    ds = ds.rename({"XLAT": "lat", "XLONG": "lon"})
    ds = ds.drop(['wspd_wdir'])

    interp_method = 'bilinear'
    ds_out = xr.Dataset({'lat': (['lat'], lats_out), 'lon': (['lon'], lons_out)})
    regridder = xe.Regridder(ds, ds_out, interp_method)
    regridder.clean_weight_file()
    ds = regridder(ds)
    ds = ds.squeeze()

    #accrain = ds.rain.rolling(Time=6, center=True).sum()
    #acclatent = ds.latent.rolling(Time=6, center=True).sum()
    #ds = ds.rolling(Time=6, center=True).mean()
    accrain = ds.rain.rolling(Time=12, center=False).sum()
    acclatent = ds.latent.rolling(Time=12, center=False).sum()
    #ds = ds.rolling(Time=6, center=True).mean()
    ds['accrain'] = accrain
    ds['acclatent'] = acclatent

    ds.to_netcdf(fout)

    return ds


def metpy_temp_adv(fname, plevels, tidx):

    ds = xr.open_dataset(fname).metpy.parse_cf().squeeze()
    ds = ds.isel(Time=tidx)
    print(ds)

    dx, dy = mpcalc.lat_lon_grid_deltas(ds.lon.values * units('degrees_E'), 
            ds.lat.values * units('degrees_N'))

    plevels_unit = plevels * units.hPa

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
    z,u,v,w,tk,th,eth,wspd,omega = log_interpolate_1d(plevels_unit,p,z,u,v,w,tk,th,eth,wspd,omega,axis=0)
    
    coords, dims = [plevs,ds.lat.values,ds.lon.values], ["level","lat","lon"]
    for name, var in zip(['z','u','v','w','tk','th','eth','wspd','omega'], [z,u,v,w,tk,th,eth,wspd,omega]):
        #g = ndimage.gaussian_filter(var, sigma=3, order=0)
        ds1[name] = xr.DataArray(to_np(var), coords=coords, dims=dims)

    # Calculate temperature advection using metpy function
    for i, plev in enumerate(plevs):

        uqvect, vqvect = mpcalc.q_vector(u[i,:,:], v[i,:,:], th[i,:,:], plev*units.hPa, dx, dy)
        #uqvect, vqvect = mpcalc.q_vector(u[i,:,:], v[i,:,:], th.to('degC')[i,:,:], plev*units.hPa, dx, dy)
        q_div = -2* mpcalc.divergence(uqvect, vqvect, dx, dy,  dim_order='yx')
        ds1['uq_{:03d}'.format(plev)] = xr.DataArray(np.array(uqvect), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
        ds1['vq_{:03d}'.format(plev)] = xr.DataArray(np.array(vqvect), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
        ds1['q_div_{:03d}'.format(plev)] = xr.DataArray(np.array(q_div), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
        
#        adv = mpcalc.advection(tk[i,:,:], [u[i,:,:], v[i,:,:]], (dx, dy), dim_order='yx') * units('K/sec')
#        adv = ndimage.gaussian_filter(adv, sigma=3, order=0) * units('K/sec')
#        ds1['tk_adv_{:03d}'.format(plev)] = xr.DataArray(np.array(adv), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
#
#        adv = mpcalc.advection(th[i,:,:], [u[i,:,:], v[i,:,:]], (dx, dy), dim_order='yx') * units('K/sec')
#        adv = ndimage.gaussian_filter(adv, sigma=3, order=0) * units('K/sec')
#        ds1['th_adv_{:03d}'.format(plev)] = xr.DataArray(np.array(adv), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
#
        adv = mpcalc.advection(eth[i,:,:], [u[i,:,:], v[i,:,:]], (dx, dy), dim_order='yx') * units('K/sec')
        adv = ndimage.gaussian_filter(adv, sigma=3, order=0) * units('K/sec')
        ds1['eth_adv_{:03d}'.format(plev)] = xr.DataArray(np.array(adv), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

        div = mpcalc.divergence(u[i,:,:], v[i,:,:], dx, dy, dim_order='yx')
        div = ndimage.gaussian_filter(div, sigma=3, order=0) * units('1/sec')
        ds1['div_{:03d}'.format(plev)] = xr.DataArray(np.array(div), coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

    ds1['Time'] = ds.Time
    eth2 = mpcalc.equivalent_potential_temperature(ds.slp.values*units.hPa, ds.t2m.values*units('K'), ds.td2.values*units('celsius'))
    ds1['eth2'] = xr.DataArray(eth2, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['sst'] = xr.DataArray(ndimage.gaussian_filter(ds.sst.values, sigma=3, order=0)-273.15, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['t2m'] = xr.DataArray(ds.t2m.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['th2'] = xr.DataArray(ds.th2.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['mask'] = xr.DataArray(ds.mask.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

    ds1['u10'] = xr.DataArray(ds.u10.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['v10'] = xr.DataArray(ds.v10.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['slp'] = xr.DataArray(ds.slp.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['rain'] = xr.DataArray(ds.rain.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    ds1['accrain'] = xr.DataArray(ds.accrain.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    #ds1['latent'] = xr.DataArray(ds.latent.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])
    #ds1['acclatent'] = xr.DataArray(ds.acclatent.values, coords=[ds.lat.values,ds.lon.values], dims=["lat","lon"])

    return ds1


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
high = cm.YlOrRd(np.linspace(0.2, 0.95, 128))
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
#drywet.set_under = 'darkbrown'
#drywet.set_over = 'magenta'


fwrf1 = "../outputs/wrf3roms1/1-1-2/wrfout_d01_2019-09-06_00:00:00"
fwrf2 = "../outputs/wrf3roms1_adj_Qs/1-1-2/wrfout_d01_2019-09-06_00:00:00"

minLon, maxLon, minLat, maxLat = [119,130,27,40]
hres = 0.1
lons = np.arange(minLon,maxLon+hres,hres) # analysis domain c    overing [118.5, 128, 26.5, 42.]
lats = np.arange(minLat,maxLat+hres,hres)

fname1 = 'data/metpy_cpl_nodown.nc'
fname2 = 'data/metpy_cpl_down.nc'

plevs = [925, 900, 850, 700, 500, 300]
#_ = metpy_read_wrf(fwrf1, fname1, lons, lats)
#_ = metpy_read_wrf(fwrf2, fname2, lons, lats)

#for tidx in np.arange(6,67,6):
tidx = 54


# Create the figure and grid for subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True, subplot_kw=dict(projection=ccrs.PlateCarree()))

# Plot 700 hPa
xtitles = ['Sep 07 at 12Z (+36hrs)','Sep 08 at 00Z (+48hrs)','Sep 08 at 12Z (+60hrs)']
fs = ['a','b','c','d','e','f']

obs_lat, obs_lon = [ 32.12295277777780, 125.182447222222 ]

j, k = 0, 0
for i, ax in enumerate(axes.flatten()):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, color='k', zorder=1)
    ax.set_extent([minLon, maxLon, minLat, maxLat])

    ax.plot(obs_lon, obs_lat, color='k', ms=10, marker='*', zorder=100, alpha=0.5 )

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=':')
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_left = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if i<len(xtitles): 
        ax.text(x=0.5, y=1.1, s=xtitles[i], fontsize=13, ha='center',va='bottom', transform=ax.transAxes)

    if i%len(xtitles)==0: 
        gl.ylabels_left = True

    ax.text(x=0.08, y=0.875, s=fs[i], fontsize=16, ha='center',va='bottom', weight='bold', transform=ax.transAxes)


axes[0,0].text(x=-0.25, y=0.5, s='CPL_down', fontsize=13, rotation=90, ha='center',va='center', transform=axes[0,0].transAxes)
axes[1,0].text(x=-0.25, y=0.5, s=r'CPL_down$-$CPL_nodown', fontsize=13, rotation=90, ha='center',va='center', transform=axes[1,0].transAxes)

for i, tidx in enumerate([36,48,60]):

    ds1 = metpy_temp_adv(fname1, plevs, tidx)
    ds2 = metpy_temp_adv(fname2, plevs, tidx)
    ds3 = ds2-ds1

    plev = 850 

    ax = axes[0,i]

    cf = ax.contourf(ds2.lon, ds2.lat, mpcalc.smooth_n_point(ds2.accrain.values, 9),
            clevs, cmap=cmap, norm=norm, extend='max')
    #cf = ax.contourf(ds2.lon, ds2.lat, ndimage.gaussian_filter(ds2.accrain.values, sigma=3, order=0), 
    #        clevs, cmap=cmap, norm=norm, extend='max')

    #cs = ax.contour(ds2.lon, ds2.lat, mpcalc.smooth_n_point(ds2.z.sel(level=plev).values/9.8, 9), 
    #        np.arange(1300,1580,10), colors='grey', linewidths=1)
    #ax.clabel(cs, fontsize=7, inline=1, inline_spacing=3, fmt='%im')

    fmin = np.min(mpcalc.smooth_n_point(ds2.accrain.values, 9))
    fmax = np.max(mpcalc.smooth_n_point(ds2.accrain.values, 9))
    ax.set_title('Min/Max = {:0.1f}/{:0.1f} mm'.format(np.array(fmin), np.array(fmax)), fontsize=10)
    #fmin = np.min(ndimage.gaussian_filter(ds2.accrain.values, sigma=3, order=0))
    #fmax = np.max(ndimage.gaussian_filter(ds2.accrain.values, sigma=3, order=0))

    skip = slice(None, None, 10)
    q = ax.quiver(ds2.lon.values[skip], ds2.lat.values[skip], ds2.u.sel(level=plev).values[skip,skip], ds2.v.sel(level=plev).values[skip,skip],
            headwidth=7, width=0.0045, scale=200, zorder=11, color='k')

    r1= patches.Rectangle((119.725,27.55), 1.8, 1.4, facecolor='w', edgecolor='k', zorder=10)
    ax.add_patch(r1)
    ax.quiverkey(q, X=0.15, Y=0.12, U=20, label='20 m/s', labelpos='S', fontproperties={'size': 'small'}, zorder=20)

    if i==len(xtitles)-1:
        cb = fig.colorbar(cf, ax=axes[0,:], format='%i', shrink=0.9, aspect=40, ticks=clevs)
        cb.set_label('12-hrs acc. precip. [mm]', fontsize=11, rotation=-90, va='bottom')
        cb.ax.tick_params(labelsize=9.5) 
        #ax.quiverkey(q, X=1.15, Y=1.08, U=20, label='20 m/s', labelpos='S', fontproperties={'size': 'small'})


    ax = axes[1,i]

    cf = ax.contourf(ds2.lon, ds2.lat, mpcalc.smooth_n_point(ds3.accrain.values, 9),
            np.arange(-22,26,4), cmap=drywet, extend='both')
            #np.arange(-20,24,4), cmap=drywet, extend='both')
    #cf = ax.contourf(ds2.lon, ds2.lat, ndimage.gaussian_filter(ds3.accrain.values, sigma=3, order=0),
    #        np.arange(-10,12,2), cmap=drywet, extend='both')

    fmin = np.min(mpcalc.smooth_n_point(ds3.accrain.values, 9))
    fmax = np.max(mpcalc.smooth_n_point(ds3.accrain.values, 9))
    ax.set_title('Min/Max = {:0.1f}/{:0.1f} mm'.format(np.array(fmin), np.array(fmax)), fontsize=10)
    #fmin = np.min(ndimage.gaussian_filter(ds3.accrain.values, sigma=3, order=0))
    #fmax = np.max(ndimage.gaussian_filter(ds3.accrain.values, sigma=3, order=0))
    #ax.set_title('Min/Max = {:0.1f}/{:0.1f} mm'.format(fmin, fmax), fontsize=10)

    #cs = ax.contour(ds2.lon, ds2.lat, mpcalc.smooth_n_point(ds3.z.sel(level=plev).values/9.8, 9), 
    #        10, colors='grey', linewidths=1)
    #ax.clabel(cs, fontsize=7, inline=1, inline_spacing=3, fmt='%im')

    skip = slice(None, None, 10)
    q = ax.quiver(ds2.lon.values[skip], ds2.lat.values[skip], ds3.u.sel(level=plev).values[skip,skip], ds3.v.sel(level=plev).values[skip,skip],
            headwidth=7, width=0.0045, scale=50, zorder=11, color='k')

    r1= patches.Rectangle((119.725,27.55), 1.8, 1.4, facecolor='w', edgecolor='k', zorder=10)
    ax.add_patch(r1)
    ax.quiverkey(q, X=0.15, Y=0.12, U=4, label='4 m/s', labelpos='S', fontproperties={'size': 'small'}, zorder=20)

    if i==len(xtitles)-1:
        cb = fig.colorbar(cf, ax=axes[1,:], shrink=0.9, aspect=40, format='%i', ticks=np.arange(-20,24,4))
        cb.set_label('Diff. of acc. precip. [mm]', fontsize=11, rotation=-90, va='bottom')
        cb.ax.tick_params(labelsize=9.5) 
        

        #ax.quiverkey(q, X=1.15, Y=1.08, U=4, label='4 m/s', labelpos='S', fontproperties={'size': 'small'})
        cb.ax.text(x=-0.25, y=1.05, s='wet', fontsize=10, transform=cb.ax.transAxes)
        cb.ax.text(x=-0.25, y=-0.08, s='dry', fontsize=10, transform=cb.ax.transAxes)


plt.show()
#plt.suptitle('Precipitation [shaded]; 850-hPa HGT [conoutrs] & winds [vecotrs] ', fontsize=16)

fig.savefig('fig_06_precip_diff.png'.format(tidx), dpi=500, bbox_inches='tight')
