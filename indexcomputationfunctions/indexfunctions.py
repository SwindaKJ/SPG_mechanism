#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:48:09 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

Functions to compute the indices in the SPG mechanism.

"""

#%% Import modules and functions

import os
import numpy as np
import xarray as xr
import gsw

from indexcomputationfunctions.indexdatafunctions import  time_to_dayssince
    
#%% INDEX FUNCTIONS

def regional_mean(var_cs, area_cs, lat_bnd, lon_bnd, func="mean"):
    """
    A function to compute the mean of a variable over a given region weighted
    by the area file.

    Parameters
    ----------
    var_cs : DataArray of the relevant variable, either tos or ua.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    lat_bnd : The latitude bounds of the region. 
        Given as [slice, south_bound, north_bound]
    lon_bnd : The longitude bounds of the region. 
        Given as [slice_360, slice_180, west_bound_360,east_bound_360, 
                  west_bound_180,east_bound_180]
    func : The function to be used (mean, max). The default is 'mean'.

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean of the variable over the 
                    selected region.

    """
    
    # Convert all times to 'days since' for consistency
    var_cs = time_to_dayssince(var_cs)
    
    # Rename variables
    if 'tos' in var_cs.variables:
        var_name = 'tos'
        var_cs = var_cs.rename(tos="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'sos' in var_cs.variables:
        var_name = 'sos'
        var_cs = var_cs.rename(sos="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'rho' in var_cs.variables:
        var_name = 'rho'
        var_cs = var_cs.rename(rho="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'zos' in var_cs.variables:
        var_name = 'zos'
        var_cs = var_cs.rename(zos="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'mlotst' in var_cs.variables:
        var_name = 'mlotst'
        var_cs = var_cs.rename(mlotst="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'msftbarot' in var_cs.variables:
        var_name = 'msftbarot'
        var_cs = var_cs.rename(msftbarot="vr")
        area_cs = area_cs.rename(areacello="areacell")
    
    print("Renamed variables")
    
    # For nearly all models
    if 'y' in var_cs.coords:
        # Check whether grids match
        if not (any(var_cs.x == area_cs.x) and any(var_cs.y == area_cs.y)):
            print("Coordinates mismatch")
            if len(var_cs.x) == len(area_cs.x) and len(var_cs.x) == len(area_cs.x):
                area_cs['x'] = var_cs['x']
                area_cs['y'] = var_cs['y']
            else:
                print("Panic! Grids don't match in size.")
        
        print("Select region")
        # Select the tradewinds over the chosen region
        if area_cs.lon.min() >= 0: #0-360 longitude convention
            var_reg = var_cs.where(var_cs.lat < lat_bnd[2]).\
                where(var_cs.lat > lat_bnd[1]).\
                where(var_cs.lon < lon_bnd[3]).\
                where(var_cs.lon > lon_bnd[2])
            area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                where(area_cs.lat > lat_bnd[1]).\
                where(area_cs.lon < lon_bnd[3]).\
                where(area_cs.lon > lon_bnd[2])
        else: # -180-180 longitude convention
            var_reg = var_cs.where(var_cs.lat < lat_bnd[2]).\
                where(var_cs.lat > lat_bnd[1]).\
                where(var_cs.lon < lon_bnd[5]).\
                where(var_cs.lon > lon_bnd[4])
            area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                where(area_cs.lat > lat_bnd[1]).\
                where(area_cs.lon < lon_bnd[5]).\
                where(area_cs.lon > lon_bnd[4])
        
        if func == "mean":
            # Computed weighted mean over region
            print("Compute weighted average")
            var_reg_wmean = var_reg.vr.weighted(area_reg.areacell.fillna(0)).\
                mean(['y','x'], skipna=True)
        elif func == "max":
            # Computed maximum over region
            print("Compute regional maximum")
            var_reg_wmean = var_reg.vr.max(['y','x'], skipna=True)
    
    elif var_cs.attrs['parent_source_id'] == 'CNRM-CM6-1-HR':
        print(var_cs.attrs['parent_source_id'])
        
        # Loop over time in steps of 10
        for t in range(int(len(var_cs.time)/10)):
            # Select time frame
            var_t = var_cs.isel(time=slice(10*t,10*(t+1)))
            # print(t)
         
            # Select region
            if area_cs.lon.min() >= 0: #0-360 longitude convention
                var_reg = var_cs.where(var_t.lat < lat_bnd[2]).\
                    where(var_t.lat > lat_bnd[1]).\
                    where(var_t.lon < lon_bnd[3]).\
                    where(var_t.lon > lon_bnd[2])
                area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                    where(area_cs.lat > lat_bnd[1]).\
                    where(area_cs.lon < lon_bnd[3]).\
                    where(area_cs.lon > lon_bnd[2])
            else: # -180-180 longitude convention
                var_reg = var_cs.where(var_t.lat < lat_bnd[2]).\
                    where(var_t.lat > lat_bnd[1]).\
                    where(var_t.lon < lon_bnd[5]).\
                    where(var_t.lon > lon_bnd[4])
                area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                    where(area_cs.lat > lat_bnd[1]).\
                    where(area_cs.lon < lon_bnd[5]).\
                    where(area_cs.lon > lon_bnd[4])
         
            if func == "mean":
                # Computed weighted mean over region
                print("Compute weighted average")
                if t == 0: # First step
                    var_reg_wmean = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                        mean(['i','j'], skipna=True)
                else: # Append
                    var_reg_done = var_reg_wmean.copy()
                    var_reg_new = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                        mean(['i','j'], skipna=True)
                    var_reg_wmean = xr.concat([var_reg_done,var_reg_new], dim='time')
            elif func == "max":
                # Computed maximum over region
                print("Compute regional maximum")
                if t == 0: # First step
                    var_reg_wmean = var_reg.vr.max(['i','j'], skipna=True)
                else: # Append
                    var_reg_done = var_reg_wmean.copy()
                    var_reg_new = var_reg.vr.max(['i','j'], skipna=True)
                    var_reg_wmean = xr.concat([var_reg_done,var_reg_new], dim='time')

    # ICON and CMCC Model
    else:
        print("Else")
        print("Select region")
        if area_cs.lon.min() >= 0: #0-360 longitude convention
            var_reg = var_cs.where(var_cs.lat < lat_bnd[2]).\
                where(var_cs.lat > lat_bnd[1]).\
                where(var_cs.lon < lon_bnd[3]).\
                where(var_cs.lon > lon_bnd[2])
            area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                where(area_cs.lat > lat_bnd[1]).\
                where(area_cs.lon < lon_bnd[3]).\
                where(area_cs.lon > lon_bnd[2])
        else: # -180-180 longitude convention
            var_reg = var_cs.where(var_cs.lat < lat_bnd[2]).\
                where(var_cs.lat > lat_bnd[1]).\
                where(var_cs.lon < lon_bnd[5]).\
                where(var_cs.lon > lon_bnd[4])
            area_reg = area_cs.where(area_cs.lat < lat_bnd[2]).\
                where(area_cs.lat > lat_bnd[1]).\
                where(area_cs.lon < lon_bnd[5]).\
                where(area_cs.lon > lon_bnd[4])
        
        if func == "mean":
            # Computed weighted mean over region
            print("Compute weighted average")
            if 'x' in var_cs.coords: # ICON
                var_reg_wmean = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                    mean('x', skipna=True)
            else: # CMCC
                var_reg_wmean = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                    mean(['i','j'], skipna=True)
        elif func == "max":
            # Computed maximum over region
            print("Compute regional maximum")
            if 'x' in var_cs.coords: # ICON
                var_reg_wmean = var_reg.vr.max(['x'], skipna=True)
            else: # CMCC
                var_reg_wmean = var_reg.vr.max(['i','j'], skipna=True)
    
    # Rename to original variable
    var_reg_wmean.name = var_name
    
    return var_reg_wmean


# SEA SURFACE TEMPERATURE
def sstatlspg_index(var_cs, area_cs, dom='std'):
    """
    Compute the weighted mean sea surface temperature over a selected region.

    Parameters
    ----------
    var_cs : DataArray containing the sea surface temperature.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.
        

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean sea surface temperature over 
                    the selected region.

    """
    # Get lat-lon bounds
    if dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd)


# SEA SURFACE SALINITY
def sssatlspg_index(var_cs, area_cs, dom='std'):
    """
    Compute the weighted mean sea surface salinity over a selected region.

    Parameters
    ----------
    var_cs : DataArray containing the sea surface temperature.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean sea surface temperature over 
                    the selected region.

    """
    # Get lat-lon bounds
    if dom == 'swingedouw': # subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    if dom == 'fingerprint': # northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd)


# MIXED LAYER DEPTH
def mldatlspg_index(var_cs, area_cs, dom='std'):
    """
    Compute the weighted mean mixed layer depth over the west of the subpolar
    gyre in the north Atlantic.

    Parameters
    ----------
    var_cs : DataArray containing the mixed layer depth.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', 'old', with the option of adding more. 
        The default is'std'.

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean mixed layer depth over the 
                    selected region.

    """
    # Get lat-lon bounds
    if dom == 'old': # Domain with initial MLD results
        lat_bnd = [slice(50,60), 50,60]
        lon_bnd = [slice(300,330), slice(-60,-30), 300,330, -60,-30]
    elif dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd)


# SEA SURFACE HEIGHT
def sshatlspg_index(var_cs, area_cs, dom='std'):
    """
    Compute the weighted mean sea surface height over the west of the subpolar
    gyre in the north Atlantic.

    Parameters
    ----------
    var_cs : DataArray containing the sea surface height.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', 'old', with the option of adding more. 
        The default is'std'.

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean sea surface height over the 
                    selected region.

    """
    # Get lat-lon bounds
    if dom == 'old': # Domain with initial MLD results
        lat_bnd = [slice(50,60), 50,60]
        lon_bnd = [slice(300,330), slice(-60,-30), 300,330, -60,-30]
    elif dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
        
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd)


# SEA SURFACE DENSITY
def rhoatlspg_index(var_cs, area_cs, dom='std'):
    """
    Compute the weighted mean sea surface temperature over a selected region.

    Parameters
    ----------
    var_cs : DataArray containing the sea surface temperature.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.
        

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean sea surface temperature over 
                    the selected region.

    """
    
    # Compute density anomaly
    rho_ar = gsw.density.rho_t_exact(var_cs[1].sos, var_cs[0].tos, 0) - 1000
    # To Dataset
    rho_cs = rho_ar.to_dataset().rename(sos='rho')
    
    # Get lat-lon bounds
    if dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean(rho_cs, area_cs, lat_bnd, lon_bnd)


# SUBPOLAR GYRE STREAMFUNCTION
def maxsfspg_index(var_cs, area_cs, dom):
    """
    Compute the maximum barotropic streamfunction over the subpolar gyre 
    in the north Atlantic.

    Parameters
    ----------
    var_cs : DataArray containing the barotropic streamfunction.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    dom : Name of the domain to be used. The default is 'std'.

    Returns
    -------
    var_reg_wmean : DataArray of the maximum barotropic streamfunction over 
                    the selected region.

    """
    # Get lat-lon bounds
    if dom == 'old': # Domain with initial results
        lat_bnd = [slice(45,64), 45,64]
        lon_bnd = [slice(300,360), slice(-60,0), 300,360, -60,0]
    elif dom == 'std': # Domain selected based on multi-model SF checks
        lat_bnd = [slice(50,65), 50,65]
        lon_bnd = [slice(300,340), slice(-60,-20), 300,340, -60,-20]
    elif dom == 'stdmld': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
        
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd, func="max")
    
def avsfspg_index(var_cs, area_cs, dom):
    """
    Compute the weighted mean barotropic streamfunction over the subpolar gyre 
    in the north Atlantic.

    Parameters
    ----------
    var_cs : DataArray containing the barotropic streamfunction.
    area_cs : DataArray containing the area corresponding to each gridpoint.

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean barotropic streamfunction 
                    over the selected region.

    """
    # Get lat-lon bounds
    if dom == 'old': # Domain with initial results
        lat_bnd = [slice(45,64), 45,64]
        lon_bnd = [slice(300,360), slice(-60,0), 300,360, -60,0]
    elif dom == 'std': # Domain selected based on multi-model SF checks
        lat_bnd = [slice(50,65), 50,65]
        lon_bnd = [slice(300,340), slice(-60,-20), 300,340, -60,-20]
    elif dom == 'stdmld': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
        
    return regional_mean(var_cs, area_cs, lat_bnd, lon_bnd)


#%% 3D VARIABLES

def regional_mean_3D(var_cs, area_cs, lat_bnd, lon_bnd, lev_bnd):
    """
    A function to compute the mean of a variable over a given region and depth
    weighted by the volume file.

    Parameters
    ----------
    var_cs : DataArray of the relevant variable, either tos or ua.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    lat_bnd : The latitude bounds of the region. 
        Given as [slice, south_bound, north_bound]
    lon_bnd : The longitude bounds of the region. 
        Given as [slice_360, slice_180, west_bound_360,east_bound_360, 
                  west_bound_180,east_bound_180]
    lev_bnd : The depth bounds to be used. Given as [top, bottom].

    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean of the variable over the 
                    selected region.

    """
    
    # Convert all times to 'days since' for consistency
    var_cs = time_to_dayssince(var_cs)
    
    # Rename variables
    if 'thetao' in var_cs.variables:
        var_name = 'thetao'
        var_cs = var_cs.rename(thetao="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'so' in var_cs.variables:
        var_name = 'so'
        var_cs = var_cs.rename(so="vr")
        area_cs = area_cs.rename(areacello="areacell")
    elif 'rho' in var_cs.variables:
        var_name = 'rho'
        var_cs = var_cs.rename(rho="vr")
        area_cs = area_cs.rename(areacello="areacell")
    print("Renamed variables")
    print(var_cs)
    
    # Get an array for th depth levels and their differences
    lev_ar = np.array(var_cs.lev)
    diff_lev = np.array(lev_ar[1::]) - np.array(lev_ar[0:-1])
    # Go from the depth-grid to weights
    lev_ar_top = np.append([0], lev_ar[0:-1]+diff_lev/2)
    lev_ar_bot = np.append(lev_ar[0:-1]+diff_lev/2, lev_ar[-1])
    lev_wght = lev_ar_bot - lev_ar_top

    # Expand the area-data with the depth-weights
    area_exp = area_cs.expand_dims(dim={"lev": lev_ar})
    area_wght = area_exp * lev_wght[:,np.newaxis,np.newaxis]
    print("Expanded area file with level weights")
    
    # Get dimensions over which to compute weighted average
    dim_listall = list(area_wght.dims)
    dim_list = [d for d in dim_listall if d in ['x','y','i','j','lev']]
    print("Dimensions:")
    print(dim_list)
    
    # Check whether grids match
    print("Grid check")
    if 'y' in var_cs.coords:
        if not (any(var_cs.x == area_wght.x) and any(var_cs.y == area_wght.y)):
            if len(var_cs.x) == len(area_wght.x) \
            and len(var_cs.x) == len(area_wght.x):
                print("Coordinates mismatch")
                area_wght['x'] = var_cs['x']
                area_wght['y'] = var_cs['y']
                
    # For nearly all models
    if not var_cs.attrs['parent_source_id'] == 'CNRM-CM6-1-HR':
        print("Select region")
        # Select the fields over the chosen region end depth (for var)
        var_reg = var_cs.where(var_cs.lat < lat_bnd[2]).\
            where(var_cs.lat > lat_bnd[1]).\
            where(var_cs.lon < lon_bnd[5]).\
            where(var_cs.lon > lon_bnd[4]).\
            where(var_cs.lev < lev_bnd[1]).\
            where(var_cs.lev > lev_bnd[0])
        
        # Computed weighted mean over region
        print("Compute weighted average")
        var_reg_wmean = var_reg.vr.weighted(area_wght.areacell.fillna(0)).\
            mean(dim_list, skipna=True)
    
    # CNRM-CM6-1-HR is too big so crashes if not split
    elif var_cs.attrs['parent_source_id'] == 'CNRM-CM6-1-HR':
        print(var_cs.attrs['parent_source_id'])
        
        print("Select region")
        # Loop over time in steps of 10
        for t in range(int(len(var_cs.time)/10)):
            # Select time frame
            var_t = var_cs.isel(time=slice(10*t,10*(t+1)))
         
            # Select the fields over the chosen region end depth (for var)
            var_reg = var_t.where(var_t.lat < lat_bnd[2]).\
                where(var_t.lat > lat_bnd[1]).\
                where(var_t.lon < lon_bnd[5]).\
                where(var_t.lon > lon_bnd[4]).\
                where(var_t.lev < lev_bnd[1]).\
                where(var_t.lev > lev_bnd[0])

            # Computed weighted mean over region
            print("Compute weighted average")
            if t == 0: # First step
                var_reg_wmean = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                    mean(dim_list, skipna=True)
            else: # Append
                var_reg_done = var_reg_wmean.copy()
                var_reg_new = var_reg.vr.weighted(area_cs.areacell.fillna(0)).\
                    mean(dim_list, skipna=True)
                var_reg_wmean = xr.concat([var_reg_done,var_reg_new], dim='time')
    
    # Rename to original variable
    var_reg_wmean.name = var_name
    
    return var_reg_wmean

# OCEAN POTENTIAL TEMPERATURE
def subthspg_index(var_cs, area_cs, lev_bnd=[50,1000], dom='std'):
    """
    Compute the weighted mean ocean potential temperature over a selected 
    region and depth.

    Parameters
    ----------
    var_cs : DataArray containing the ocean potential temperature
    area_cs : DataArray containing the area corresponding to each gridpoint.
    lev_bnd : The depth-bounds to consider. The default is [50,1000]
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.
        
    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean ocean potential temperature.
    
    """
    # Get lat-lon bounds
    if dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean_3D(var_cs, area_cs, lat_bnd, lon_bnd, lev_bnd)

# OCEAN SALINITY
def subsospg_index(var_cs, area_cs, lev_bnd=[50,1000], dom='std'):
    """
    Compute the weighted mean ocean salinity over a selected region and depth.

    Parameters
    ----------
    var_cs : DataArray containing the ocean salinity.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    lev_bnd : The depth-bounds to consider. The default is [50,1000]
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.
        
    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean ocean salinity.

    """
    # Get lat-lon bounds
    if dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean_3D(var_cs, area_cs, lat_bnd, lon_bnd, lev_bnd)


# OCEAN DENSITY AT DEPTH
def subrhospg_index(var_cs, area_cs, lev_bnd=[50,1000], dom='std'):
    """
    Compute the weighted mean ocean density over a selected region and depth.

    Parameters
    ----------
    var_cs : DataArray containing the ocean salinity.
    area_cs : DataArray containing the area corresponding to each gridpoint.
    lev_bnd : The depth-bounds to consider. The default is [50,1000]
    dom : The domain to be used. Options for now are 'std' (my domain), 
        'swingedouw', 'fingerprint', with the option of adding more. 
        The default is'std'.
        
    Returns
    -------
    var_reg_wmean : DataArray of the weighted mean ocean density.

    """
    
    # Compute density anomaly
    rho_ar = gsw.density.rho_t_exact(var_cs[1].so, var_cs[0].thetao, 0) - 1000
    # To Dataset
    rho_cs = rho_ar.to_dataset().rename(so='rho')
    
    # Get lat-lon bounds
    if dom == 'swingedouw': # Subpolar gyre region (Swingedouw et al. 2021)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(290,340), slice(-70,-20), 290,340, -70,-20]
    elif dom == 'fingerprint': # Northern fingerprint region (Caesar et al. 2018)
        lat_bnd = [slice(45,60), 45,60]
        lon_bnd = [slice(315,340), slice(-45,-20), 315,340, -45,-20]
    elif dom == 'std': # Domain selected based on multi-model MLD checks
        lat_bnd = [slice(54,63), 54,63]
        lon_bnd = [slice(300,313), slice(-60,-47), 300,313, -60,-47]
    return regional_mean_3D(rho_cs, area_cs, lat_bnd, lon_bnd, lev_bnd)










