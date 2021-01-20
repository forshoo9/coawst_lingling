import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
from copy import copy

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, ALL_TIMES,
                 cartopy_ylim, latlon_coords, interplevel, extract_vars,
                 omp_enabled, omp_get_num_procs, omp_set_num_threads,
                 extract_times, ll_to_xy)

from netCDF4 import Dataset, num2date, MFDataset

from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import NearestNDInterpolator, interp2d, griddata
import seawater as sw
from scipy.signal._peak_finding import _boolrelextrema

from scipy.stats import norm, linregress, pearsonr

def wrf_interp2d(points, Z, xi, Tindex):
    z = to_np(Z)
    name = Tindex.name
    da = xr.DataArray(dims=[name], coords={name: Tindex})
    for t in range(np.shape(z)[0]):
        da[t] = gd(points, z[t,:,:].flatten(), xi)[0]
    return (da)


def get_oml_depth_total(netcdf_file_path, ntim):
    nc = Dataset(netcdf_file_path)
    mask_rho = nc.variables['mask_rho'][:]
    x_slice0 = slice(0, mask_rho.shape[1])
    y_slice0 = slice(0, mask_rho.shape[0])

    mld_out = []
    for t_in in range(ntim):
        t_slice = slice(t_in, t_in+1)
        tyxSlicesArray = [t_slice, y_slice0, x_slice0]
    
        z = compute_depths(netcdf_file_path, tyxSlicesArray)
        temp_profile = nc.variables['temp'][t_slice,:, y_slice0, x_slice0]
        salt_profile = nc.variables['salt'][t_slice,:, y_slice0, x_slice0]
        z = z.squeeze()
        temp_profile = temp_profile.squeeze()
        salt_profile = salt_profile.squeeze()
        
        sigma_profile = sw.dens(salt_profile, temp_profile, 0)-1000
        mld, _, _, _, _, _ = compute_mld_based_on_density_curvature( 
                sigma_profile, z, qi_treshold=0.55, )

        mld_out.append(mld.filled(np.nan))

    return mld_out


def get_oml_depth(netcdf_file_path, t_in):
    nc = Dataset(netcdf_file_path)
    mask_rho = nc.variables['mask_rho'][:]
    x_slice0 = slice(0, mask_rho.shape[1])
    y_slice0 = slice(0, mask_rho.shape[0])
    t_slice = slice(t_in, t_in+1)
    tyxSlicesArray = [t_slice, y_slice0, x_slice0]

    z = compute_depths(netcdf_file_path, tyxSlicesArray)
    temp_profile = nc.variables['temp'][t_slice,:, y_slice0, x_slice0]
    salt_profile = nc.variables['salt'][t_slice,:, y_slice0, x_slice0]
    z = z.squeeze()
    temp_profile = temp_profile.squeeze()
    salt_profile = salt_profile.squeeze()
    
    sigma_profile = sw.dens(salt_profile, temp_profile, 0)-1000
    mld, _, _, _, _, _ = compute_mld_based_on_density_curvature( 
            sigma_profile, z, qi_treshold=0.55, )

    return mld


def ekman_pumping(curl_tau, f, rho_water=1028., tdim=2):
    """Calculate Ekman pumping from wind-stress curl.

    Args:
        curl_tau: Wind stress curl (N/m^3), 2d or 3d (for time series).
        y: Latitude grid (degrees), 2d.

    Notes:
        We = Curl(tau)/rho*f (vertical velocity in m/s).
        f = Coriolis frequency (rad/s), latitude dependent.
        rho = Ocean water density (1028 kg/m^3).

    """
    ## Coriolis frequency
    #omega = 7.292115e-5 # rotation rate of the Earth (rad/s)
    #f = 2 * omega * np.sin(y * np.pi/180) # (rad/s)
    #print(f)

    ## Expand dimension for broadcasting (2d -> 3d)
    if f.shape != curl_tau.shape:
        f = np.expand_dims(f, tdim)

    # Ekman pumping
    We = curl_tau / (rho_water * f) # vertical velocity (m/s)
    return We


def get_plot_element(infile):
    rootgroup = Dataset(infile, 'r')
    p = getvar(rootgroup, 'RAINNC')
    lats, lons = latlon_coords(p)
    cart_proj = get_cartopy(p)
    xlim = cartopy_xlim(p)
    ylim = cartopy_ylim(p)
    rootgroup.close()
    return cart_proj, xlim, ylim, lons, lats

class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

def get_roms_nc_parameters(nc_file_path):
    netcdf_dataset_impl = Dataset
    if ( type(nc_file_path) is list):
        netcdf_dataset_impl = MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    Vtransform = nc['Vtransform'][:]
    Vstretching = nc['Vstretching'][:]
    sc_r = nc['s_rho'][:]
    Cs_r = nc['Cs_r'][:]
    sc_w = nc['s_w'][:]
    Cs_w = nc['Cs_w'][:]
    return Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w

def shear(z, u, v=0):
    r"""
    Calculates the vertical shear for u, v velocity section.
    .. math::
        \\textrm{shear} = \\frac{\\partial (u^2 + v^2)^{0.5}}{\partial z}
    Parameters
    ----------
    z : array_like
        depth [m]
    u(z) : array_like
           Eastward velocity [m s :sup:`-1`]
    v(z) : array_like
           Northward velocity [m s :sup:`-1`]
    Returns
    -------
    shr : array_like
          frequency [s :sup:`-1`]
    z_ave : array_like
            depth between z grid (M-1xN)  [m]
    Examples
    --------
    >>> import oceans.sw_extras.sw_extras as swe
    >>> z = [[0], [250], [500], [1000]]
    >>> u = [[0.5, 0.5, 0.5], [0.15, 0.15, 0.15],
    ...      [0.03, 0.03, .03], [0.,0.,0.]]
    >>> swe.shear(z, u)[0]
    array([[-1.4e-03, -1.4e-03, -1.4e-03],
           [-4.8e-04, -4.8e-04, -4.8e-04],
           [-6.0e-05, -6.0e-05, -6.0e-05]])
    """
    z, u, v = list(map(np.asanyarray, (z, u, v)))
    z, u, v = np.broadcast_arrays(z, u, v)

    m = z.shape[0]
    #m, n = z.shape
    iup = np.arange(0, m - 1)
    ilo = np.arange(1, m)
    z_ave = (z[iup] + z[ilo]) / 2.0
    vel = np.sqrt(u ** 2 + v ** 2)
    diff_vel = np.diff(vel, axis=0)
    diff_z = np.diff(z, axis=0)
    shr = diff_vel / diff_z
    return shr, z_ave

def compute_depths(nc_file_path, tyxSlices, igrid=1, idims=0):
    Vtransform, Vstretching, sc_r, Cs_r, sc_w, Cs_w = get_roms_nc_parameters(nc_file_path)
    netcdf_dataset_impl = Dataset
    if (type(nc_file_path) is list):
        netcdf_dataset_impl = MFDataset
    nc = netcdf_dataset_impl(nc_file_path)
    # Read in S-coordinate parameters.
    N = len(sc_r)
    Np = N + 1

    if (len(sc_w) == N):
        sc_w = np.cat(-1, sc_w.transpose())
        Cs_w = np.cat(-1, Cs_w.transpose())

    # Get bottom topography.
    yxSlice = [tyxSlices[1], tyxSlices[2]]
    h = nc.variables['h'][yxSlice]
    [Mp, Lp] = h.shape
    L = Lp - 1
    M = Mp - 1

    # Get free-surface
    zeta = nc.variables['zeta'][tyxSlices]
    # zeta=np.zeros([Lp, Mp])

    if igrid == 1:
        if idims == 1:
            h = h.transpose()
            zeta = zeta.transpose()
    elif igrid == 2:
        hp = 0.25 * (h[1:L, 1:M] + h[2:Lp, 1:M] + h[1:L, 2:Mp] + h[2:Lp, 2:Mp])
        zetap = 0.25 * (zeta[1:L, 1:M] + zeta[2:Lp, 1:M] + zeta[1:L, 2:Mp] + zeta[2:Lp, 2:Mp])
        if idims:
            hp = hp.transpose()
            zetap = zetap.transpose()
    elif igrid == 3:
        hu = 0.5 * (h[1:L, 1:Mp] + h[2:Lp, 1:Mp])
        zetau = 0.5 * (zeta[1:L, 1:Mp] + zeta[2:Lp, 1:Mp])
        if idims:
            hu = hu.transpose()
            zetau = zetau.transpose()
    elif igrid == 4:
        hv = 0.5 * (h[1:Lp, 1:M] + h[1:Lp, 2:Mp])
        zetav = 0.5 * (zeta[1:Lp, 1:M] + zeta[1:Lp, 2:Mp])
        if idims:
            hv = hv.transpose()
            zetav = zetav.transpose()
    elif igrid == 5:
        if idims:
            h = h.transpose()
            zeta = zeta.transpose()

    # Set critical depth parameter.
    hc = np.min(h[:])
    if 'hc' in nc.variables:
        hc = nc['hc'][:]
    
    # Compute depths, for a different variables size will not match
    if (Vtransform == 1):
        if igrid == 1:
            for k in range(N):
                z0 = (sc_r - Cs_r) * hc + Cs_r(k) * h
                z = z0 + zeta * (1.0 + z0 / h)

    elif Vtransform == 2:
        if igrid == 1:
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * 
                    Cs_r[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 4:
            S = (hc * sc_r[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * 
                    Cs_r[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S
        elif igrid == 5:
            S = (hc * sc_w[np.newaxis, :, np.newaxis, np.newaxis] + h[np.newaxis, np.newaxis, :, :] * 
                    Cs_w[np.newaxis, :, np.newaxis, np.newaxis]) / (hc + h[np.newaxis, np.newaxis, :, :])
            z = zeta[:, np.newaxis, :, :] + (zeta[:, np.newaxis, :, :] + h[np.newaxis, np.newaxis, :, :]) * S

    return z

def compute_mld_based_on_density_curvature(sigma, z, qi_treshold=0.55, plot_diags=False):
    # z is assumed to follow roms convention, negative, from bottom to top
    # modified from the original idea by
    # Ocean mixed layer depth: A subsurface proxy of ocean-atmosphere variability, K. Lorbacher, D. Dommenget, P. P. Niiler, A. Kohl, 12 July 2006, DOI: 10.1029/2003JC002157

    sigma_sorted = sigma[::-1]
    z_sorted = -1 * z[::-1]
    dz = np.diff(z_sorted, axis=0)
    # forward difference
    sigma_gradient = np.diff(sigma_sorted, axis=0) / np.diff(z_sorted, axis=0)
    sigma_curvature = np.empty(sigma_gradient.shape)
    # backward difference
    sigma_curvature[1:] = np.diff(sigma_gradient, axis=0) / np.diff(z_sorted[:-1], axis=0)
    #at the edge have to use forward difference
    sigma_curvature[0] = np.diff(sigma_gradient[0:2], axis=0) / np.diff(z_sorted[0:2], axis=0)

    # find curvature local maxima
    curvature_extrema_max_ind = _boolrelextrema(sigma_curvature, np.greater, order=1, axis=0)
    #find positive first derivative and convex down
    ind_positive_derivative_positive_curvature = np.logical_and(sigma_gradient > 0, sigma_curvature > 0)
    #keep only correct maxima
    ind_max_extremum = np.logical_and(curvature_extrema_max_ind, ind_positive_derivative_positive_curvature)

    qi_data = np.empty(ind_max_extremum.shape)
    qi_data[:] = np.NaN

    #compute quality index everywhere
    for i in range(ind_max_extremum.shape[0]):
        qi_data[i] = 1 - np.std(sigma_sorted[0:i], axis=0) / np.std(sigma_sorted[0:int(min(i*1.5, ind_max_extremum.shape[0]))], axis=0)

    masked_qi_data = np.ma.array(qi_data, mask=np.logical_not(ind_max_extremum))

    final_ind = np.empty(sigma_sorted.shape[1:])
    final_ind[:] = np.NaN

    profile_max_qi_data = np.nanmax(masked_qi_data, axis=0)
    sub_index = profile_max_qi_data >= qi_treshold
    final_ind[sub_index] = np.nanargmax(sigma_curvature[:, sub_index], axis=0)
    # Otherwise pick local extremum/maxima which has highest qi index
    #final_ind[sub_index] = np.nanargmax(masked_qi_data[:, sub_index], axis=0)
    # if there is no local extremum or qi is weak, then pick point with max gradient value
    sub_index = np.logical_or(np.isnan(profile_max_qi_data), profile_max_qi_data < qi_treshold)
    final_ind[sub_index] = np.nanargmax(sigma_gradient[:, sub_index], axis=0)

    final_ind = final_ind.astype(int)
    fancy_indices = np.indices(ind_max_extremum.shape[1:])

    mld = np.empty(sigma_sorted.shape[1:])
    mld[:] = np.NaN
    mld = z_sorted[final_ind, fancy_indices[0], fancy_indices[1]]

    # invert in z order everything back to original and make compatible shape
    current_sigma_gradient_temp = np.zeros(sigma_sorted.shape)
    current_sigma_gradient_temp[:-1] = sigma_gradient
    current_sigma_gradient_temp[-1] = np.NaN
    sigma_gradient = current_sigma_gradient_temp[::-1]

    current_sigma_curvature_temp = np.zeros(sigma_sorted.shape)
    current_sigma_curvature_temp[:-1] = sigma_curvature
    current_sigma_curvature_temp[-1] = np.NaN
    sigma_curvature = current_sigma_curvature_temp[::-1]

    ind_max_extremum_temp = np.zeros(sigma_sorted.shape, dtype=bool)
    ind_max_extremum_temp[:-1] = ind_max_extremum
    ind_max_extremum_temp[-1] = np.NaN
    ind_max_extremum = ind_max_extremum_temp[::-1]

    qi_data_temp = np.zeros(sigma_sorted.shape)
    qi_data_temp[:-1] = qi_data
    qi_data_temp[-1] = np.NaN
    qi_data = qi_data_temp[::-1]

    return mld, sigma_gradient, sigma_curvature, final_ind, ind_max_extremum, qi_data



def wind_stress(u, v, rho_air=1.22, cd=None):
    """Convert wind speed (u,v) to wind stress (Tx,Ty).

    It uses either wind-dependent or constant drag.

    Args:
        u, v: Wind vector components (m/s), 2d or 3d (for time series).
        rho_air: Density of air (1.22 kg/m^3).
        cd: Non-dimensional drag (wind-speed dependent).
            For constant drag use cd=1.5e-3.
    Notes:
        Function to compute wind stress from wind field data is based on Gill,
        (1982)[1]. Formula and a non-linear drag coefficient (cd) based on
        Large and Pond (1981)[2], modified for low wind speeds (Trenberth et
        al., 1990)[3]

        [1] A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press.
        [2] W.G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate
        to Strong Winds, J. Physical Oceanography, v11, p324-336.
        [3] K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual
        Cycle in Global Ocean Wind Stress, J. Physical Oceanography, v20,
        p1742-1760.

    Fernando Paolo <fpaolo@ucsd.edu>
    Mar 7, 2016

    """
    w = np.sqrt(u**2 + v**2) # wind speed (m/s) 

    if not cd:
        # wind-dependent drag
        cond1 = (w<=1)
        cond2 = (w>1) & (w<=3)
        cond3 = (w>3) & (w<10)
        cond4 = (w>=10)
        cd = np.zeros_like(w)
        cd[cond1] = 2.18e-3 
        cd[cond2] = (0.62 + 1.56/w[cond2]) * 1e-3
        cd[cond3] = 1.14e-3
        cd[cond4] = (0.49 + 0.065*w[cond4]) * 1e-3

    Tx = rho_air * cd * w * u # zonal wind stress (N/m^2)
    Ty = rho_air * cd * w * v # meridional wind stress (N/m^2)
    return [Tx, Ty]


def divergence(Tx, Ty, dx, dy, ydim=0, xdim=1, tdim=2):
    """Calculate the curl of wind stress (Tx, Ty).

    Args:
        Tx, Ty: Wind stress components (N/m^2), 2d or 3d (for time series)
        x, y: Coordinates in lon/lat (degrees), 2d.

    Notes:
        Curl(Tx,Ty) = dTy/dx - dTx/dy
        The different constants come from oblateness of the ellipsoid.

    """
    #dy = np.abs(y[1,0] - y[0,0]) # scalar in deg
    #dx = np.abs(x[0,1] - x[0,0]) 
    #dy *= 110575. # scalar in m
    #dx *= 111303. * np.cos(y * np.pi/180) # array in m (varies w/lat)

    # extend dimension for broadcasting (2d -> 3d)
    #if Tx.ndim == 3:
    #    dx = np.ones(np.shape(Tx))*dx
    #    dy = np.ones(np.shape(Tx))*dy
    #    #dx = np.expand_dims(dx, tdim)

        
    # grad[f(y,x), delta] = diff[f(y)]/delta, diff[f(x)]/delta 
    dTxdy = np.gradient(Tx, dy)[ydim] # (N/m^3)
    dTydx = np.gradient(Ty, dx)[xdim] 
    div_tau = dTydx + dTxdy # (N/m^3)
    return div_tau


def wind_stress_curl(Tx, Ty, dx, dy, ydim=0, xdim=1, tdim=2):
    """Calculate the curl of wind stress (Tx, Ty).

    Args:
        Tx, Ty: Wind stress components (N/m^2), 2d or 3d (for time series)
        x, y: Coordinates in lon/lat (degrees), 2d.

    Notes:
        Curl(Tx,Ty) = dTy/dx - dTx/dy
        The different constants come from oblateness of the ellipsoid.

    """
    #dy = np.abs(y[1,0] - y[0,0]) # scalar in deg
    #dx = np.abs(x[0,1] - x[0,0]) 
    #dy *= 110575. # scalar in m
    #dx *= 111303. * np.cos(y * np.pi/180) # array in m (varies w/lat)

    # extend dimension for broadcasting (2d -> 3d)
    #if Tx.ndim == 3:
    #    dx = np.ones(np.shape(Tx))*dx
    #    dy = np.ones(np.shape(Tx))*dy
    #    #dx = np.expand_dims(dx, tdim)

        
    # grad[f(y,x), delta] = diff[f(y)]/delta, diff[f(x)]/delta 
    dTxdy = np.gradient(Tx, dy)[ydim] # (N/m^3)
    dTydx = np.gradient(Ty, dx)[xdim] 
    curl_tau = dTydx - dTxdy # (N/m^3)
    return curl_tau


def regline(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    corr = pearsonr(x,y)
    rms = np.sqrt(np.square(np.subtract(x, y)).mean())
    mae = np.mean(np.abs(np.subtract(x,y)))
    mape = np.mean(np.abs(np.subtract(x,y)/x)) * 100.
    nrms = rms / (np.subtract(max(x),min(x))) * 100.
    return(slope)

def bearing(lon1, lat1, lon2, lat2):
    d2r = np.pi/180.
    r2d = (1/d2r)
 
    lat1r, lon1r = lat1*d2r, lon1*d2r
    lat2r, lon2r = lat2*d2r, lon2*d2r
   
    ang_tmp = r2d*np.arctan2(np.sin((lon2r-lon1r))*np.cos(lat2r),
            np.cos(lat1r)*np.sin(lat2r)-np.sin(lat1r)*np.cos(lat2r)*np.cos(lat2r-lat1r))
    ang = (ang_tmp+360)%360
 
    return(ang)
 
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    

def lambert_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])


def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels


def atleast_1d(*arrs):
    r"""Convert inputs to arrays with at least one dimension.

    Scalars are converted to 1-dimensional arrays, whilst other
    higher-dimensional inputs are preserved. This is a thin wrapper
    around `numpy.atleast_1d` to preserve units.

    Parameters
    ----------
    arrs : arbitrary positional arguments
        Input arrays to be converted if necessary

    Returns
    -------
    `pint.Quantity`
        A single quantity or a list of quantities, matching the number of inputs.

    """
    mags = [a.magnitude if hasattr(a, 'magnitude') else a for a in arrs]
    orig_units = [a.units if hasattr(a, 'units') else None for a in arrs]
    ret = np.atleast_1d(*mags)
    if len(mags) == 1:
        if orig_units[0] is not None:
            return units.Quantity(ret, orig_units[0])
        else:
            return ret
    return [units.Quantity(m, u) if u is not None else m for m, u in zip(ret, orig_units)]


def _broadcast_to_axis(arr, axis, ndim):
    """Handle reshaping coordinate array to have proper dimensionality.

    This puts the values along the specified axis.
    """
    if arr.ndim == 1 and arr.ndim < ndim:
        new_shape = [1] * ndim
        new_shape[axis] = arr.size
        arr = arr.reshape(*new_shape)
    return arr


def _process_deriv_args(f, kwargs):
    """Handle common processing of arguments for derivative functions."""
    n = f.ndim

    from numpy.core.numeric import normalize_axis_index
    axis = normalize_axis_index(kwargs.get('axis', 0), n)

    if f.shape[axis] < 3:
        raise ValueError('f must have at least 3 point along the desired axis.')

    if 'delta' in kwargs:
        if 'x' in kwargs:
            raise ValueError('Cannot specify both "x" and "delta".')

        delta = atleast_1d(kwargs['delta'])
        if delta.size == 1:
            diff_size = list(f.shape)
            diff_size[axis] -= 1
            delta_units = getattr(delta, 'units', None)
            delta = np.broadcast_to(delta, diff_size, subok=True)
            if not hasattr(delta, 'units') and delta_units is not None:
                delta = delta * delta_units
        else:
            delta = _broadcast_to_axis(delta, axis, n)
    elif 'x' in kwargs:
        x = _broadcast_to_axis(kwargs['x'], axis, n)
        delta = diff(x, axis=axis)
    else:
        raise ValueError('Must specify either "x" or "delta" for value positions.')

    return n, axis, delta

def concatenate(arrs, axis=0):
    r"""Concatenate multiple values into a new unitized object.

    This is essentially a unit-aware version of `numpy.concatenate`. All items
    must be able to be converted to the same units. If an item has no units, it will be given
    those of the rest of the collection, without conversion. The first units found in the
    arguments is used as the final output units.

    Parameters
    ----------
    arrs : Sequence of arrays
        The items to be joined together

    axis : integer, optional
        The array axis along which to join the arrays. Defaults to 0 (the first dimension)

    Returns
    -------
    `pint.Quantity`
        New container with the value passed in and units corresponding to the first item.

    """
    dest = 'dimensionless'
    for a in arrs:
        if hasattr(a, 'units'):
            dest = a.units
            break

    data = []
    for a in arrs:
        if hasattr(a, 'to'):
            a = a.to(dest).magnitude
        data.append(np.atleast_1d(a))

    # Use masked array concatenate to ensure masks are preserved, but convert to an
    # array if there are no masked values.
    data = np.ma.concatenate(data, axis=axis)
    if not np.any(data.mask):
        data = np.asarray(data)

    return units.Quantity(data, dest)

def diff(x, **kwargs):
    """Calculate the n-th discrete difference along given axis.

    Wraps :func:`numpy.diff` to handle units.

    Parameters
    ----------
    x : array-like
        Input data
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as that of the input.

    See Also
    --------
    numpy.diff

    """
    if hasattr(x, 'units'):
        ret = np.diff(x.magnitude, **kwargs)
        # Can't just use units because of how things like temperature work
        it = x.flat
        true_units = (next(it) - next(it)).units
        return true_units * ret
    else:
        return np.diff(x, **kwargs)

def first_derivative(f, x, axis): #**kwargs):
    """Calculate the first derivative of a grid of values.

    Works for both regularly-spaced data and grids with varying spacing.

    Either `x` or `delta` must be specified, or `f` must be given as an `xarray.DataArray` with
    attached coordinate and projection information. If `f` is an `xarray.DataArray`, and `x` or
    `delta` are given, `f` will be converted to a `pint.Quantity` and the derivative returned
    as a `pint.Quantity`, otherwise, if neither `x` nor `delta` are given, the attached
    coordinate information belonging to `axis` will be used and the derivative will be returned
    as an `xarray.DataArray`.

    This uses 3 points to calculate the derivative, using forward or backward at the edges of
    the grid as appropriate, and centered elsewhere. The irregular spacing is handled
    explicitly, using the formulation as specified by [Bowen2005]_.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axis : int or str, optional
        The array axis along which to take the derivative. If `f` is ndarray-like, must be an
        integer. If `f` is a `DataArray`, can be a string (referring to either the coordinate
        dimension name or the axis type) or integer (referring to axis number), unless using
        implicit conversion to `pint.Quantity`, in which case it must be an integer. Defaults
        to 0.
    x : array-like, optional
        The coordinate values corresponding to the grid points in `f`.
    delta : array-like, optional
        Spacing between the grid points in `f`. Should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The first derivative calculated along the selected axis.

    See Also
    --------
    second_derivative

    """
    n = f.ndim
    delta = np.diff(x, axis=axis)

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * n
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    delta_slice0 = [slice(None)] * n
    delta_slice1 = [slice(None)] * n

    # First handle centered case
    slice0[axis] = slice(None, -2)
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    delta_slice0[axis] = slice(None, -1)
    delta_slice1[axis] = slice(1, None)

    combined_delta = delta[tuple(delta_slice0)] + delta[tuple(delta_slice1)]
    delta_diff = delta[tuple(delta_slice1)] - delta[tuple(delta_slice0)]
    center = (- delta[tuple(delta_slice1)] / (combined_delta * delta[tuple(delta_slice0)])
              * f[tuple(slice0)]
              + delta_diff / (delta[tuple(delta_slice0)] * delta[tuple(delta_slice1)])
              * f[tuple(slice1)]
              + delta[tuple(delta_slice0)] / (combined_delta * delta[tuple(delta_slice1)])
              * f[tuple(slice2)])

#    # Fill in "left" edge with forward difference
#    slice0[axis] = slice(None, 1)
#    slice1[axis] = slice(1, 2)
#    slice2[axis] = slice(2, 3)
#    delta_slice0[axis] = slice(None, 1)
#    delta_slice1[axis] = slice(1, 2)
#
#    combined_delta = delta[tuple(delta_slice0)] + delta[tuple(delta_slice1)]
#    big_delta = combined_delta + delta[tuple(delta_slice0)]
#    left = (- big_delta / (combined_delta * delta[tuple(delta_slice0)])
#            * f[tuple(slice0)]
#            + combined_delta / (delta[tuple(delta_slice0)] * delta[tuple(delta_slice1)])
#            * f[tuple(slice1)]
#            - delta[tuple(delta_slice0)] / (combined_delta * delta[tuple(delta_slice1)])
#            * f[tuple(slice2)])
#
#    # Now the "right" edge with backward difference
#    slice0[axis] = slice(-3, -2)
#    slice1[axis] = slice(-2, -1)
#    slice2[axis] = slice(-1, None)
#    delta_slice0[axis] = slice(-2, -1)
#    delta_slice1[axis] = slice(-1, None)
#
#    combined_delta = delta[tuple(delta_slice0)] + delta[tuple(delta_slice1)]
#    big_delta = combined_delta + delta[tuple(delta_slice1)]
#    right = (delta[tuple(delta_slice1)] / (combined_delta * delta[tuple(delta_slice0)])
#             * f[tuple(slice0)]
#             - combined_delta / (delta[tuple(delta_slice0)] * delta[tuple(delta_slice1)])
#             * f[tuple(slice1)]
#             + big_delta / (combined_delta * delta[tuple(delta_slice1)])
#             * f[tuple(slice2)])

    #return concatenate((left, center, right), axis=axis)
    return center

