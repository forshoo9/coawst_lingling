import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

from metpy.cbook import get_test_data
from metpy.interpolate import log_interpolate_1d
from metpy.plots import add_metpy_logo, add_timestamp
from metpy.units import units
import scipy.ndimage as ndimage

import xesmf as xe
import numpy as np
import xarray as xr
import pandas as pd

import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from netCDF4 import Dataset, num2date, MFDataset

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, ALL_TIMES,
                 cartopy_ylim, latlon_coords, interplevel, extract_vars,
                 omp_enabled, omp_get_num_procs, omp_set_num_threads,
                 extract_times, ll_to_xy, get_basemap)

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.path as mpath

from yang_func import *

def get_hurricane():
    u = np.array([  [2.444,7.553],
                    [0.513,7.046],
                    [-1.243,5.433],
                    [-2.353,2.975],
                    [-2.578,0.092],
                    [-2.075,-1.795],
                    [-0.336,-2.870],
                    [2.609,-2.016]  ])
    u[:,0] -= 0.098
    codes = [1] + [2]*(len(u)-2) + [2] 
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpath.Path(3*u, codes, closed=False)


low = cm.GnBu_r(np.linspace(0.01,0.9, 128))
mid = np.ones((50,4))
high = cm.YlOrRd(np.linspace(0.1, 0.95, 128))
colors = np.vstack((low, mid, high))
bwr = LinearSegmentedColormap.from_list('my_colormap', colors)#, N=24)
bwr.set_over = 'darkbrown'
bwr.set_under = 'magemta'
bwr.set_bad = 'k'

plevs = [950,850,500,300,200]

def metpy_read_wrf(tidx, froms, froms0, fwrf, lons_out, lats_out):

    wrfin = Dataset(fwrf)
    
    ds = xr.Dataset()

    slp = getvar(wrfin, "slp", timeidx=tidx)
    ds['slp'] = smooth2d(slp, 3, cenweight=4)

    lats, lons = latlon_coords(slp)
    ds['lats'] = lats
    ds['lons'] = lons
    landmask = extract_vars(wrfin, timeidx=tidx, varnames=["LANDMASK"] ).get("LANDMASK")
    u10 = extract_vars(wrfin, timeidx=tidx, varnames=["U10"] ).get("U10")
    v10 = extract_vars(wrfin, timeidx=tidx, varnames=["V10"] ).get("V10")
    ds['u10'] = u10.where(landmask==0)
    ds['v10'] = v10.where(landmask==0)
    latent = extract_vars(wrfin, timeidx=tidx, varnames=["LH"] ).get("LH")
    ds['latent'] = smooth2d(latent, 3, cenweight=4)#latent.where(landmask==0)
    t2m = extract_vars(wrfin, timeidx=tidx, varnames=["T2"] ).get("T2")-273.15
    ds['t2m'] = smooth2d(t2m, 3, cenweight=4) 
    sst = extract_vars(wrfin, timeidx=tidx, varnames=["SST"] ).get("SST")
    ds['sst'] = sst.where(landmask==0)

    romsin = xr.open_dataset(froms)
    romsin = romsin.rename({"lat_rho": "lat", "lon_rho": "lon"})
    romsin = romsin.isel(ocean_time=tidx)

    ds['sst_5m'] = romsin.isel(z_r=0).temp
    ds['water_temp'] = romsin.temp
    ds['water_ucur'] = romsin.ucur/100
    ds['water_vcur'] = romsin.vcur/100
    ds.water_ucur.attrs['units'] = 'm/s'
    ds.water_vcur.attrs['units'] = 'm/s'
 
    romsin = xr.open_dataset(froms0)
    romsin = romsin.rename({"lat_rho": "lat", "lon_rho": "lon"})
    romsin = romsin.isel(ocean_time=tidx)
    ds['h'] = romsin.h
   
    mld = get_oml_depth(froms0, t_in=tidx)
    mld = smooth2d(mld, 3, cenweight=4) 
    ds['oml_depth'] = xr.DataArray(mld, ds.sst_5m.coords, ds.sst_5m.dims, ds.sst_5m.attrs)
    ds['oml_depth2'] = ds.oml_depth.where(ds.h>20)

    ds = ds.drop(['XLONG','XLAT','XTIME','Time'])
    ds = ds.rename({'south_north': 'eta_rho', 'west_east': 'xi_rho'})

    interp_method = 'bilinear'
    ds_out = xr.Dataset({'lat': (['lat'], lats_out), 'lon': (['lon'], lons_out)})
    regridder = xe.Regridder(ds, ds_out, interp_method)
    regridder.clean_weight_file()
    ds = regridder(ds)
    ds = ds.squeeze()

    dxy=10000.

    ds = ds.metpy.parse_cf().squeeze()

    utau, vtau = wind_stress(to_np(ds.u10), to_np(ds.v10))
    ds['u_tau'] = xr.DataArray(utau, ds.u10.coords, ds.u10.dims, ds.u10.attrs)
    ds['v_tau'] = xr.DataArray(vtau, ds.v10.coords, ds.v10.dims, ds.v10.attrs)
    curl = mpcalc.vorticity(utau * units('m/s'), utau * units('m/s'), 
                dx=dxy* units.meter, dy=dxy* units.meter)
    ds['wind_stress_curl'] = xr.DataArray(np.array(curl), ds.u10.coords, ds.u10.dims, ds.u10.attrs)
    
    div = []
    for z in range(len(ds.z_r)):
        div0 = mpcalc.divergence(ds.water_ucur.isel(z_r=z)* units('m/s'),
                ds.water_vcur.isel(z_r=z)* units('m/s'), 
                dx=dxy* units.meter, dy=dxy* units.meter)
        div.append(div0)
    ds['cur_div'] = xr.DataArray(div, ds.water_ucur.coords, ds.water_ucur.dims, ds.water_ucur.attrs)

    div = mpcalc.divergence(ds.water_ucur.isel(z_r=2)* units('m/s'),
            ds.water_vcur.isel(z_r=2)* units('m/s'), 
            dx=dxy* units.meter, dy=dxy* units.meter)
    ds['surf_cur_div'] = xr.DataArray(np.array(div), ds.u10.coords, ds.u10.dims, ds.u10.attrs)

    print(ds)
    return ds


tf = pd.read_csv('storm_center_lingling_case_total_adj_Qs.csv', index_col='date', parse_dates=True)

df0 = tf[tf.cases=='ATM_MO']
df1 = tf[tf.cases=='CPL_MO']
df2 = tf[tf.cases=='CPL_MO2']

fig, axes = plt.subplots(2, 3, figsize=(9, 8), constrained_layout=True,
        subplot_kw={'projection':ccrs.PlateCarree()}, sharex=True, sharey=True)

fs = ['a','b','c','d','e','f']

lons = np.arange(118.5,128.1,0.1) # analysis domain covering [118.5, 128, 26.5, 42.]
lats = np.arange(26.5,42.1,0.1)
    
fwrf = "../outputs/wrf3roms1_adj_Qs/1-1-2/wrfout_d01_2019-09-06_00:00:00"
froms = '../outputs/wrf3roms1_adj_Qs/1-1-2/Lingling_ocean_his_standard_zlevs_under_100m.nc'
froms0 = '../outputs/wrf3roms1_adj_Qs/1-1-2/Lingling_ocean_his.nc'
ds0 = metpy_read_wrf(0, froms, froms0, fwrf, lons, lats)
weights = np.cos(np.deg2rad(ds0.lats))

row=[0,0,0,1,1,1,]
col=[0,1,2,0,1,2,]

#for i, tidx in enumerate(np.arange(6,7,6)):
for i, tidx in enumerate(np.arange(6,42,6)):

    if tidx <= 36: 
        center_lon, center_lat = df2.lon[tidx], df2.lat[tidx]

    ds = metpy_read_wrf(tidx, froms, froms0, fwrf, lons, lats)

    print(i, tidx)

    ax = axes[row[i], col[i]]

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=':',
            #xlocs=[118,120,122,124,126,128,130], auto_inline=False,
            xformatter=LONGITUDE_FORMATTER, yformatter=LATITUDE_FORMATTER)
    gl.xpadding=20
    gl.ypadding=10

    gl.rotate_labels = False
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS['land'], edgecolor='none', linewidth=0.5)
    
    ax.plot(df2.lon, df2.lat, color='red', lw=2, zorder=10)
    
    delta_sst = ds.sst-273.15-ds0.sst_5m
    mesh = ax.contourf(ds.lon, ds.lat, delta_sst.values, np.arange(-7,8,1), cmap=bwr, extend='both')

#    delta_oml = ds.oml_depth-ds0.oml_depth
#    cs = ax.contour(ds.lon, ds.lat, mpcalc.smooth_n_point(delta_oml.values, 9), [-40,-30,-20,-10,10,20,30,40], colors='k', linewidths=1)
#    ax.clabel(cs, fontsize=9, inline=1, inline_spacing=3, fmt='%i')
##
#    cs = ax.contour(ds.lon, ds.lat, -ds.latent.values, [10], colors='lime', linewidths=2)
#    cs = ax.contour(ds.lon, ds.lat, -ds.latent.values, [200], colors='orange', linewidths=2)
#    cs = ax.contour(ds.lon, ds.lat, -ds.latent.values, [400], colors='purple', linewidths=2)

    delta_oml = ds.oml_depth-ds0.oml_depth

    #ch = ax.contourf(ds.lon, ds.lat, delta_oml.values, [20,100000], colors='none', hatches=['..'])

    x = np.array(ds.lons.isel(lon=slice(None,None,2), lat=slice(None,None,2)).values).flatten()
    y = np.array(ds.lats.isel(lon=slice(None,None,2), lat=slice(None,None,2)).values).flatten()
    z = np.array(delta_oml.isel(lon=slice(None,None,2), lat=slice(None,None,2)).values).flatten()
    mask = np.where(z>20)
    ax.plot(x[mask], y[mask], 'k.', ms=1.5)

    cs = ax.contour(ds.lon, ds.lat, -ds.latent.values, [0, 100, 200, 300, 400], colors='k', linewidths=1)
    ax.clabel(cs, fontsize=8, inline=1, inline_spacing=0, fmt='%i')
    #cs = ax.contour(ds.lon, ds.lat, delta_oml.values, [10], colors='lime', linewidths=2)
    #cs = ax.contour(ds.lon, ds.lat, delta_oml.values, [20], colors='orange', linewidths=2)
    #cs = ax.contour(ds.lon, ds.lat, delta_oml.values, [20], colors='purple', linewidths=2)
    
    wslice = slice(None,None,2)
    qc = ax.quiver(ds.lon[wslice], ds.lat[wslice], 
            ds.u_tau.where(ds.wind_stress_curl*1e5>2)[wslice,wslice], 
            ds.v_tau.where(ds.wind_stress_curl*1e5>2)[wslice,wslice], 
            color='dimgrey', width=0.005, scale=80, zorder=11)

    ax.grid(True, ls=':', )
    ax.set_extent([118.5, 127.5, 28, 41])

    if i==0: 
        ax.text(0.35, 0.5, 'TC track', fontsize=11, color='r', transform=ax.transAxes)
        ax.text(0.775, 0.335, 'I-ORS', fontsize=11, color='k', transform=ax.transAxes)
        ax.text(0.34, 0.15, 'TC center', fontsize=11, color='r', transform=ax.transAxes)
        #ax.text(0.02, 0.95, r"(shaded) $\Delta$SST", fontsize=7, ha='left', color='k', transform=ax.transAxes)
        ax.text(0.02, 0.95, r"(dots) OML deepening > 20m", fontsize=7, ha='left', color='k', transform=ax.transAxes)
        ax.text(0.02, 0.90, r"(contours) downward Q$_{L}$", fontsize=7, ha='left', color='k', transform=ax.transAxes)
        ax.text(0.02, 0.86, r"  from 0 to 400 by 100 W/m$^2$", fontsize=7, ha='left', color='k', transform=ax.transAxes)
        ax.text(0.02, 0.81, r"(vectors) curl_$\tau$ > 2e5 s$^{-1}$", fontsize=7, ha='left', color='k', transform=ax.transAxes)
        #ax.text(0.02, 0.81, r"(dots) $\Delta$OML > 20m", fontsize=7, ha='left', color='k', transform=ax.transAxes)
    
    if tidx < 36: 
        hurricane = get_hurricane()
        ax.scatter(center_lon, center_lat, s=25, marker='o', 
                    edgecolors="w", facecolors='w', linewidth=2.5, zorder=19)
        ax.scatter(center_lon, center_lat, s=250, marker=hurricane, 
                    edgecolors="r", facecolors='none', linewidth=2.5, zorder=20)
    else:
        ax.scatter(center_lon, center_lat, s=100, marker='x', 
                    color="r", linewidth=2.5, zorder=19)
    
    ax.scatter(125.182, 32.123, s=100, c='k', marker='*', zorder=20)

    ax.set_title(ds.ocean_time.dt.strftime("%H00Z, %b %d").values, color='k', size=14);
    #ax.set_title(ds.ocean_time.dt.strftime("%HZ %d %b").values+" ({:02d}h)".format(tidx), color='k', size=14);

    ax.text(x=0.02, y=1.0075, s=fs[i], fontsize=16, weight='bold', 
        ha='center',va='bottom', transform=ax.transAxes)

cbar = fig.colorbar(mesh, ax=axes, aspect=70, shrink=0.9, orientation='horizontal')
cbar.ax.set_title(r"SST change relative to SST0 [K]", fontsize=13)
#cbar.set_label(r"The black dots and contours indicate $\Delta$OML > 20 m (i.e., deepening) and downward Q$_{L}$ from 0 to 400 by 100 W/m$^2$, respectively.", fontsize=10, loc='right')
#        #+r" ***vectors: curl of $\tau$ > 2e5 s$^{-1}$.",
cbar.ax.text(x=0, y=1.75, s='cooling', fontsize=10, ha='left', transform=cbar.ax.transAxes)
cbar.ax.text(x=1, y=1.75, s='warming', fontsize=10, ha='right', transform=cbar.ax.transAxes)
        #fontsize=13, rotation=-90, va='bottom' )
#cbar.set_label('SST change relative SST0 [K, shaded]', fontsize=13, rotation=-90, va='bottom' )
#cbar.set_label(r'$\Delta$SST [shaded, K]', fontsize=13, rotation=-90, va='bottom' )

#plt.suptitle("Composite maps")
plt.show()

fig.savefig('fig_04_sst_oml.png', dpi=500, bbox_inches='tight')

