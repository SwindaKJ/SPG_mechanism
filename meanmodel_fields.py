#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:05:06 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

Code to compute model statistics, specifically time mean and count of the mixed
layer deoth exceeding 1000m.

"""


#%% IMPORT MODULES

import os
import numpy as np
import xarray as xr
import xesmf as xe

from netCDF4 import Dataset

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from global_land_mask import globe

import xmip.preprocessing as xmip_pre

    
#%% IMPORT FUNCTIONS

# Move to the directory where the function files are stored
os.chdir('/Users/3753808/Library/CloudStorage/'
         'OneDrive-UniversiteitUtrecht/Code/Tipping_links/')

from indexcomputationfunctions.indexdatafunctions import obtain_wgetlist, \
    download_data

# Sjoerd's preprocessing wrapper
from preprocess_sjoerd import correct_lon_180
    
#%% DIRECTORIES

# Directory where wget files are stored
dir_wget = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links/CMIP6_wgetfiles/'

# Directory where to save the data
dir_save = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links/CMIP6_modelmeans/'
            
# Directory where to save the data
dir_save_mld = '/Users/3753808/Library/CloudStorage/' \
                'OneDrive-UniversiteitUtrecht/Code/Tipping_links/CMIP6_mld1000/'


#%% CHECK PREPROCESSING

def rename_latlon(var_xr):
    """
    A function to rename x and y (or i and j) to lat and lon and drop 
    dimensions that are not coordinates.

    Parameters
    ----------
    var_xr : Dataarray containing CMIP data after Sjoerd's preprocessing 
                function.

    Returns
    -------
    var_xr : Dataarray with renamed dimensions and dropped coordinates.

    """

    for crd in var_xr.coords:
        if crd not in var_xr.dims:
            var_xr = var_xr.drop(crd)
    if 'x' in var_xr.coords:
        var_xr = var_xr.rename(x='lon')
    if 'y' in var_xr.coords:
        var_xr = var_xr.rename(y='lat')
    if 'i' in var_xr.coords:
        var_xr = var_xr.rename(i='lon')
    if 'j' in var_xr.coords:
        var_xr = var_xr.rename(j='lat')
    
    return var_xr

def preprocessing_wrapper(ds):
    """
    Preprocessing wrapper to standardize input for the regridding. It also
    removes redundant coordinates that the regridding algorithm does not like.

    Parameters
    ----------
    ds : Dataset of CMIP6 data.

    Returns
    -------
    ds : Dataset after preprocessing

    """

    # fix naming
    ds = xmip_pre.rename_cmip6(ds)
    # promote empty dims to actual coordinates
    ds = xmip_pre.promote_empty_dims(ds)
    # demote coordinates from data_variables
    ds = xmip_pre.correct_coordinates(ds)
    # broadcast lon/lat
    ds = xmip_pre.broadcast_lonlat(ds)
    # shift all lons to consistent -180 - 180
    ds = correct_lon_180(ds)
    # fix the units and metadata
    ds = xmip_pre.correct_units(ds)
    ds = xmip_pre.fix_metadata(ds)
    
    # GFDL model
    if "x_deg" in ds.coords:
        ds = ds.rename({'x_deg': 'x', 'y_deg':'y'})
    
    # drop unused coordinates (all except lon, lat, lev, x, y, time, time_bnds)
    _drop_coords = [
        "bnds", "vertex", "lon_bounds", "lat_bounds", "lat_bnds", "lon_bnds", 
        "height","lat_verticies", "lon_verticies"
        ]
    ds = ds.drop_vars(_drop_coords, errors="ignore")
    
    # drop unused dimensions
    _drop_dims = [
        "verticies", "vertex", "lon_bounds", "lat_bounds", "lon_bnds", 
        "lat_bnds", "bounds", "bnds", "height"
    ]
    ds = ds.drop_dims(_drop_dims, errors="ignore")
    
    # rename time_bounds to time_bnds
    try:
        ds = ds.rename({"time_bounds": "time_bnds"})
    except ValueError:
        pass
    
    return ds

def regridding(ds):
    """
    A function to regrid the data to a .5 lat - 1 lon grid. It uses the 
    nearest_s2d method for regridding and works for both 1D and 2D grids. After
    regridding it changes the grid to 1D (which is in line with the 
    rectinlinear grid) for ease of further computation.

    Parameters
    ----------
    ds : The input CMIP6 dataset (after preprocessing).

    Returns
    -------
    var_final : The regridded dataset.

    """
    
    # Get the dimensions of the grid (rectilinear or not)
    dim_grid = len(ds.lat.shape)
    
    if dim_grid == 1: # rectilinear grid
        print("1D")
        # For ICON, drop x
        if ds.attrs['parent_source_id'] == 'ICON-ESM-LR':
            ds = ds.drop_indexes(["x"]).drop(["x"])
        # Set new grid
        ngrid = xr.Dataset(
            {"lat": (["lat"], np.arange(-89.75, 90, 0.5)),
              "lon": (["lon"], np.arange(-179.5, 180, 1.0)),})
        # Get whether the points are in the ocean
        lon_grid, lat_grid = np.meshgrid(ngrid.lon, ngrid.lat)
        landmask = globe.is_ocean(lat_grid, lon_grid)
        # Add mask variable
        newgrid = ngrid.merge(xr.Dataset({'mask':(('lat','lon'), landmask*1)}))
        # Regridding (using conservative)
        regridder = xe.Regridder(ds, newgrid, "nearest_s2d", periodic=True)
        # regridder = xe.Regridder(ds, newgrid, "bilinear")
        var_grid = regridder(ds, skipna=True)
    elif dim_grid == 2: # other grid
        print("2D")
        # Drop x and y as index coordinates
        ds = ds.drop_indexes(["x","y"]).drop(["x","y"])
        # Set new grid
        ngrid = xe.util.grid_global(1, 0.5)
        # Get whether the points are in the ocean
        landmask = globe.is_ocean(ngrid.lat, ngrid.lon)
        # Add mask variable
        newgrid = ngrid.merge(xr.Dataset({'mask':(('y','x'), landmask*1)}))
        # Regridding
        regridder = xe.Regridder(ds, newgrid, "nearest_s2d", periodic=True)
        # regridder = xe.Regridder(ds, ngrid, "bilinear", periodic=True)
        var_grid = regridder(ds, skipna=True)
    else:
        print("Check your input data for regridding.")
    
    # Get lat and lon as 1D
    var_1D = xmip_pre.replace_x_y_nominal_lat_lon(var_grid)
    # Rename x, y, i, j to lat and lon
    var_final = rename_latlon(var_1D)
    
    return var_final

def timemean_vars(file_list, dir_datafiles, region=True):
    """
    A function to compute the time mean of a CMIP6 model run. It regrids so
    that the results are on the same grid for all models.

    Parameters
    ----------
    file_list : The list of model files.
    dir_datafiles : The directory in which the files are stored.
    region : Select whether or not to only keep the data for the North Atlantic
        region. The default is True.

    Returns
    -------
    var_mean : The time mean of the CMIP6 model dataset (after regridding).

    """
    
    print(len(file_list))
    count = 0
    
    # For every file
    for filename in file_list:
        print(filename)
    
        # Create path to file
        filepath = os.path.join(dir_datafiles, filename)
        # Load dataset
        var_dataset = Dataset(filepath, mode='r')
        # Transfer to xarray
        var_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(var_dataset))
        print(var_xr)
        
        # Preprocessing
        var_prep = preprocessing_wrapper(var_xr)
        print(var_prep)
        
        # Set the new grid
        var_grid = regridding(var_prep)
        print(var_grid)
        
        # Select SPG region
        if region == True:
            var_reg = var_grid.sel(lat=slice(40,80), lon=slice(-80,20))
        else:
            var_reg = var_grid.copy()
        
        # Compute time mean for the full year
        var_mean_one = var_reg.mean(['time'], skipna=True)
        len_one = [len(var_reg.time)]
        # Compute the mean for winter
        var_mean_winter = var_reg.isel(time=(var_reg.time.dt.season == 'DJF')) \
                            .mean(['time'], skipna=True)
        # Append
        var_mean_one = xr.concat([var_mean_one, var_mean_winter], dim="period")
        # For each month
        for i in range(1,13):
            var_mean_month = var_reg.isel(time=(var_reg.time.dt.month == i)) \
                                .mean(['time'], skipna=True)
            var_mean_one = xr.concat([var_mean_one, var_mean_month], dim="period")

        # Make clear what the different periods correspond to
        var_mean_one = var_mean_one.assign_coords(period=["year", "DJF", 1,2,3,
                                                          4,5,6,7,8,9,10,11,12])
        
        # Append in time over all files
        if count == 0: # Create for first file
            var_meanf = var_mean_one.copy()
            len_files = len_one.copy()
        else: # Append the others
            var_meanf = xr.concat([var_meanf, var_mean_one.copy()], 
                                  dim="filetime")
            len_files = np.append(len_files, len_one)
        
        count+=1
    
    # Average over the different files
    if len(len_files) > 1: 
        print(len_files)
        if all(lf == len_files[0] for lf in len_files):
            var_mean = var_meanf.mean(['filetime'], skipna=True)
        else:
            print("Not the same length.")
            var_mean = var_meanf.weighted(xr.DataArray(data=len_files, 
                                                       dims='filetime'))\
                        .mean(['filetime'], skipna=True)
    else:
        var_mean = var_meanf.copy()
    
    return var_mean               


def count_mld1000(file_list, dir_datafiles, threshold=1000, region=False):
    """
    A function to count how often in each model the mixed layer depth (mlotst)
    exceeds a threshold value (1000m) over the whole run. It regrids so that 
    the results are on the same grid for all models.

    Parameters
    ----------
    file_list : The list of model files.
    dir_datafiles : The directory in which the files are stored.
    threshold : The value above which data are counted.
    region : Select whether or not to only keep the data for the North Atlantic
        region. The default is True.

    Returns
    -------
    var_count : The number of times the mixed layer depth exceeds the threshold
        for each gridcell for the CMIP6 model dataset (after regridding).

    """
    
    print(len(file_list))
    count = 0
    
    # For every file
    for filename in file_list:
        print(filename)
    
        # Create path to file
        filepath = os.path.join(dir_datafiles, filename)
        # Load dataset
        var_dataset = Dataset(filepath, mode='r')
        # Transfer to xarray
        var_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(var_dataset))
        print(var_xr)
        
        # Preprocessing
        var_prep = preprocessing_wrapper(var_xr)
        print(var_prep)
        
        # Set the new grid
        var_grid = regridding(var_prep)
        print(var_grid)
        
        # Select SPG region
        if region == True:
            var_reg = var_grid.sel(lat=slice(40,80), lon=slice(-80,20))
        else:
            var_reg = var_grid.copy()
        
        # Only keep MLD exceeding a threshold (ones, rest is zero)
        var_exc_thres = var_reg.where(var_reg.mlotst > threshold).notnull()*1.
        # Get time length
        len_one = [len(var_reg.time)]
        
        # For each year count the number of months with convection
        var_year = var_exc_thres.groupby('time.year').sum('time')
        # If larger than zero, set to one
        var_year_mld = var_year.where(var_year.mlotst > 0.).notnull()*1.
        # Count the number of years with convection in each grid cell
        var_mod1000 = var_year_mld.sum(['year'], skipna=True)
        
        # For winter (DJF)
        var_DJF = var_exc_thres.isel(time=(var_reg.time.dt.season == 'DJF')) \
                    .groupby('time.year').sum('time')
        # If larger than zero, set to one
        var_DJF_mld = var_DJF.where(var_DJF.mlotst > 0.).notnull()*1.
        # Append
        var_mod1000 = xr.concat([var_mod1000, var_DJF_mld.sum(['year'], 
                                                              skipna=True)], 
                                dim="period")
        
        # For each month
        for i in range(1,13):
            print(i)
            # Count how many times convection (MLD>1000m) occurs
            var_mod = var_exc_thres.isel(time=(var_exc_thres.time.dt.month 
                                               == i)) \
                        .sum(['time'], skipna=True)
            # Append
            var_mod1000 = xr.concat([var_mod1000, var_mod], dim="period")
        
        # Make clear what the different periods correspond to
        var_mod1000 = var_mod1000.assign_coords(period=["year", "DJF", 1,2,3,4,
                                                        5,6,7,8,9,10,11,12])
        
        # Append in time over all files
        if count == 0: # Create for first file
            var_mldf = var_mod1000.copy()
            len_files = len_one.copy()
        else: # Append the others
            var_mldf = xr.concat([var_mldf, var_mod1000.copy()], 
                                 dim="filetime")
            len_files = np.append(len_files, len_one)
        
        count+=1
    
    # Sum over the different files
    if len(len_files) > 1: 
        var_mld = var_mldf.sum(['filetime'], skipna=True)
    else:
        var_mld = var_mldf.copy()
    
    return var_mld               


#%% COMPUTE TIME MEAN OR MLD EXCEEDING 1000M FOR EACH MODEL
"""
Compute the mean model value or count how often MLD exceeds 1000m or each 
model. Only do this once and save. Everything after uses loads saved results.
"""

# Variable to consider
var = "mlotst"
compute_timemean = False
setting = "mld1000"

if compute_timemean:
    # Get list of wget files and directory
    wget_list, dir_wgetvar = obtain_wgetlist( var, "piControl", dir_wget)
    
    # For each model (wget-file)
    for file in wget_list:
        print(file)
        # Download data
        dir_datafiles, file_list = download_data("modelmean", file, 
                                                 dir_wgetvar)
        # Compute time mean (2D)
        if setting == "timemean":
            var_mean = timemean_vars(file_list, dir_datafiles, False)
        elif setting == "mld1000":
            var_mean = count_mld1000(file_list, dir_datafiles, 1000, False)
    
        # Save results
        save_dir = os.path.join(dir_save, var, 'piControl')
        # If the directory does not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save as netcdf
        var_mean.to_netcdf(os.path.join(save_dir, 
                                        file.replace('.sh','.modelmean.nc')))
        print("Saved")



#%% FUNCTION FOR MODELMEAN

def modelmean(var, per, dir_save):
    """
    A function to compute the multi-model mean for a given set of models.

    Parameters
    ----------
    var : The name of the variable to consider.
    period : The period over which the mean is computed (year, DJF, month)
    dir_save : The directory in which the data for each model are stored.

    Returns
    -------
    mod_sub : A list of the models considered for the model mean (only first
                                                                  version)
    var_reg_mean : DataArray with the data for each model.
    var_modmean : DataArray with the multi-model mean data.

    """
    
    # Set the path to the directory where the data is stored
    path_modmean = os.path.join( dir_save, var, "piControl")
    # Get a list of files in that directory
    file_list = [x for x in os.listdir( path_modmean ) if x[0] == 'C']
    file_list.sort()
    # And the corresponding models
    mod_list = np.array([filename.split('.')[1] + "_"  + filename.split('.')[3] 
                         for filename in file_list])

    # Remove models with multiple versions
    for i in range(0,len(mod_list)):
        if mod_list[i][-8::] not in ['r1i1p1f1','r1i1p1f2']:
            mod_list[i] = 0
        if i > 0 and mod_list[i][0:-9] == mod_list[i-1][0:-9]:
            mod_list[i] = 0
    mod_sub = [i for i in mod_list if i != '0']
    
    # Set count to zero for the first model
    count = 0

    # Append data for all models
    for mod in mod_sub:
        # Open dataset
        var_mean = xr.open_dataset( os.path.join(path_modmean, 
                                    np.array(file_list)[mod_list == mod][0]) )
        # Select the period (year, DJF, month) and region
        var_reg = var_mean.sel(period=per, lat=slice(40,80), lon=slice(-80,20))
        
        # For the barotropic streamfunction
        if var == "msftbarot":
            var_reg = (var_reg - var_mean.sel(period=per, lat=10.75, 
                                              lon=-16.5)) *10**(-9)
            # Change sign for some models
            if mod[0:4] in ["CMCC", "EC-E", "IPSL"]:
                var_reg = -var_reg

        # Append in over the models
        if count == 0: # Create for first file
            var_reg_mean = var_reg.copy()
        else: # Append the others
            var_reg_mean = xr.concat([var_reg_mean, var_reg.copy()], dim="model")
            
        count=+1

    # Compute model mean
    var_modmean = var_reg_mean.mean(['model'], skipna=True)
    
    return mod_sub, var_reg_mean, var_modmean

#%% FUNCTIONS FOR PLOTTING

def plot_modelmean(var, period, mod_sub, var_reg_mean, cbar_max, cbar_min, 
                   cbar_step, cbar_label, lon_bnd, lat_bnd, lins, 
                   saveplot=False):
    """
    A function to plot the model mean for a given variable over a given period
    for all models of which data is input. In addition the bounds of boxes to 
    be drawn can be given.

    Parameters
    ----------
    var : The variable to plot.
    period : The period over which the mean has been computed.
    mod_sub : The list of models available for the given variable.
    var_reg_mean : Dataset of the model means, add .[var] at the end.
    cbar_max : The maximum value to use for the colorbar.
    cbar_min : The minimum value to use for the colorbar.
    cbar_step : The stepsize to use for the colorbar.
    cbar_label : The label to use for the colorbar.
    lon_bnd : Longitude bounds of the box(es) to draw.
    lat_bnd : Latitude bounds of the box(es) to draw.
    lins : Linestyles of the box(es) to draw..
    saveplot : Set to True to save the plots. The default is False.

    Returns
    -------
    None.

    """
    
    # Compute colorbar
    cbar_nr = 2*cbar_max /cbar_step
    # cbar_tick = range( cbar_min, cbar_max+cbar_step, cbar_step )
    cbar_tick = np.arange( cbar_min, cbar_max+cbar_step, cbar_step )

    # Box settings
    linc = 'k';         linw = 1.5;

    # Load data for all models
    for i in range(len(mod_sub)):

        # Get model and model mean over region
        mod = mod_sub[i]
        var_reg = var_reg_mean.isel(model=i)
        
        fig = plt.figure(figsize=(12,7))
        
        ax = fig.add_subplot(1,1,1, 
                             projection=ccrs.Orthographic(central_longitude=320, 
                                                          central_latitude=55))
        ax.coastlines(color='tab:gray', linewidth=0.5)
        ax.gridlines(color='tab:gray', linestyle=':')
        
        # Plot data
        ax.set_title(mod, fontsize=18)
        sub = ax.contourf(var_reg.lon, var_reg.lat, var_reg, 
                          cbar_tick, transform=ccrs.PlateCarree(), 
                          cmap = plt.cm.get_cmap('bwr',cbar_nr), 
                          vmin=cbar_min, vmax=cbar_max, extend='both')
        # Set colorbar
        cbar = fig.colorbar( sub, ticks=cbar_tick )
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, fontsize=14) 
        
        # Set boxes
        for i in range(len(lon_bnd)):
            l1 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                          lat_bnd[i,0]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                          c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
            l2 = plt.plot(lon_bnd[i,0]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                          np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                          c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
            l3 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                          lat_bnd[i,1]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                          c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
            l4 = plt.plot(lon_bnd[i,1]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                          np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                          c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        
        # Save if ordered
        if saveplot == True:
            plt.savefig(dir_save+"/figs/models/"+var+"/"+period+"/"+var+"_"
                        +period+"_"+mod+".pdf")
        plt.show()
        
    return
    
def plot_multimodelmean(var, period, var_modmean, cbar_max, cbar_min, 
                        cbar_step, cbar_label, lon_bnd, lat_bnd, lins, 
                        saveplot=False):
    """
    A function to plot the multi-model mean for a given variable over a given 
    period. In addition the bounds of boxes to be drawn can be given.

    Parameters
    ----------
    var : The variable to plot.
    period : The period over which the mean has been computed.
    var_reg_mean : Dataset of the model means, add .[var] at the end.
    cbar_max : The maximum value to use for the colorbar.
    cbar_min : The minimum value to use for the colorbar.
    cbar_step : The stepsize to use for the colorbar.
    cbar_label : The label to use for the colorbar.
    lon_bnd : Longitude bounds of the box(es) to draw.
    lat_bnd : Latitude bounds of the box(es) to draw.
    lins : Linestyles of the box(es) to draw..
    saveplot : Set to True to save the plots. The default is False.

    Returns
    -------
    None.

    """
    
    # Compute colorbar
    cbar_nr = 2*cbar_max /cbar_step
    # cbar_tick = range( cbar_min, cbar_max+cbar_step, cbar_step )
    cbar_tick = np.arange( cbar_min, cbar_max+cbar_step, cbar_step )

    # Box settings
    linc = 'k';         linw = 1.5;

    fig0 = plt.figure(figsize=(12,7))

    ax0 = fig0.add_subplot(1,1,1, 
                           projection=ccrs.Orthographic( central_longitude=320, 
                                                        central_latitude=55))
    ax0.coastlines(color='tab:gray', linewidth=0.5)
    ax0.gridlines(color='tab:gray', linestyle=':')

    # Plot data
    ax0.set_title("Multi-Model Mean", fontsize=18)
    sub0 = ax0.contourf(var_modmean.lon, var_modmean.lat, var_modmean, 
                       cbar_tick, transform=ccrs.PlateCarree(), 
                       cmap = plt.cm.get_cmap('bwr',cbar_nr), 
                       vmin=cbar_min, vmax=cbar_max, extend='both')
    # Set colorbar
    cbar = fig0.colorbar( sub0, ticks=cbar_tick )
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=14) 

    # Set boxes
    for i in range(len(lon_bnd)):
        l1 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                      lat_bnd[i,0]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l2 = plt.plot(lon_bnd[i,0]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                      np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l3 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                      lat_bnd[i,1]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l4 = plt.plot(lon_bnd[i,1]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                      np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())

    # Save if ordered
    if saveplot == True:
        plt.savefig(dir_save+"/figs/"+var+"_"+period+"_modelmean.pdf")
    plt.show()
    
    return

#%% PLOT SETTINGS

# Box settings
lon_bnd = np.array([[300,313]])
lat_bnd = np.array([[54,63]])
lins = ['-']


#%% LOAD AND PLOT MIXED LAYER DEPTH

# Variable and period to consider
var = "mlotst"
period = "DJF"
cbar_label = "Mixed Layer Depth (m)"

# Compute multi-model mean
mod_sub, var_reg_mean, var_modmean = modelmean(var, period, dir_save)

# Get the settings for the colorbar
cbar_max = 700;      cbar_min = 0;       cbar_step = 50
# Plot the multi-model mean
plot_multimodelmean(var, period, var_modmean.mlotst, cbar_max, cbar_min, 
                    cbar_step, cbar_label, lon_bnd, lat_bnd, lins, True)


#%% LOAD MLD EXCEEDENCE DATA

# Variable to consider
var = "mlotst"
per = "DJF"
mld_bound = 100

# Set the path to the directory where the data is stored
path_modmean = os.path.join( dir_save_mld, var, "piControl")
# Get a list of files in that directory
file_list = [x for x in os.listdir( path_modmean ) if x[0] == 'C']
file_list.sort()
# And the corresponding models
mod_list = np.array([filename.split('.')[1] + "_"  + filename.split('.')[3] 
                     for filename in file_list])

# Remove models with multiple versions
for i in range(0,len(mod_list)):
    if mod_list[i][-8::] not in ['r1i1p1f1','r1i1p1f2']:
        mod_list[i] = 0
    if i > 0 and mod_list[i][0:-9] == mod_list[i-1][0:-9]:
        mod_list[i] = 0
mod_sub = [i for i in mod_list if i != '0']

# Set count to zero for the first model
count = 0

# Append data for all models
for mod in mod_sub:
    # Open dataset
    var_mean = xr.open_dataset( os.path.join(path_modmean, 
                                np.array(file_list)[mod_list == mod][0]) )
    # Select the period (year, DJF, month) and region
    var_reg = var_mean.sel(period=per, lat=slice(40,80), lon=slice(-80,20))

    # Append in over the models
    if count == 0: # Create for first file
        var_reg_all = var_reg.copy()
    else: # Append the others
        var_reg_all = xr.concat([var_reg_all, var_reg.copy()], dim="model")
        
    count=+1

# Only keep gridpoints above zero
var_reg_mean = var_reg_all.where(var_reg_all.mlotst > 0.0)

# Compute model mean
# var_modmean_count = var_reg_mean.count(dim='model')
var_modmean_count = var_reg_mean.where(var_reg_mean.mlotst > mld_bound)\
                        .count(dim='model')
# Only keep gridpoints above zero
var_modmean = var_modmean_count.where(var_modmean_count.mlotst > 0.0)

#%% PLOT FOR EACH MODEL

cbar_label = "Number of years with MLD > 1000m"
saveplot=False

cbar_max = 500;      cbar_min = 0;       cbar_step = 25
# Compute colorbar
cbar_nr = 2*cbar_max /cbar_step
# cbar_tick = range( cbar_min, cbar_max+cbar_step, cbar_step )
cbar_tick = np.arange( cbar_min, cbar_max+cbar_step, cbar_step )

# Box settings
linc = 'k';         linw = 1.5;

# Load data for all models
for i in range(len(mod_sub)):

    # Get model and model mean over region
    mod = mod_sub[i]
    var_reg = var_reg_mean.isel(model=i)
    
    fig = plt.figure(figsize=(12,7))
    
    ax = fig.add_subplot(1,1,1, 
                         projection=ccrs.Orthographic(central_longitude=320, 
                                                      central_latitude=55))
    ax.coastlines(color='tab:gray', linewidth=0.5)
    ax.gridlines(color='tab:gray', linestyle=':')
    
    # Plot data
    ax.set_title(mod, fontsize=18)
    sub = ax.contourf(var_reg.lon, var_reg.lat, var_reg.mlotst, 
                      cbar_tick, transform=ccrs.PlateCarree(), 
                      cmap = plt.cm.get_cmap('bwr',cbar_nr), 
                      vmin=cbar_min, vmax=cbar_max, extend='both')
    # Set colorbar
    cbar = fig.colorbar( sub, ticks=cbar_tick )
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=14) 
    
    # Set boxes
    for i in range(len(lon_bnd)):
        l1 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                      lat_bnd[i,0]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l2 = plt.plot(lon_bnd[i,0]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                      np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l3 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                      lat_bnd[i,1]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
        l4 = plt.plot(lon_bnd[i,1]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                      np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                      c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
    
    # Save if ordered
    if saveplot == True:
        plt.savefig(dir_save+"/figs/models/mld1000/"+per+"/"+var+"_"
                    +per+"_"+mod+"_mld1000.pdf")
    plt.show()
    
    
#%% PLOT NUMBER OF MODELS EXCEEDING WITH MLD EXCEEDING 1000M (PAPER)
    
cbar_label = "Number of models"
saveplot=False

cbar_max = 24;      cbar_min = 0;       cbar_step = 2
# Compute colorbar
cbar_nr = 2*cbar_max /cbar_step
# cbar_tick = range( cbar_min, cbar_max+cbar_step, cbar_step )
cbar_tick = np.arange( cbar_min, cbar_max+cbar_step, cbar_step )

# Box settings
linc = 'k';         linw = 3;

fig0 = plt.figure(figsize=(12,11))

ax0 = fig0.add_subplot(1,1,1, 
                       projection=ccrs.Orthographic( central_longitude=320, 
                                                    central_latitude=55))
ax0.coastlines(color='tab:gray', linewidth=0.5)
ax0.gridlines(color='tab:gray', linestyle=':')

# Plot data
ax0.set_title("MLD > 1000m in at least "+repr(mld_bound)+" model years", 
              fontsize=24)
sub0 = ax0.contourf(var_modmean.lon, var_modmean.lat, var_modmean.mlotst, 
                    cbar_tick, transform=ccrs.PlateCarree(), 
                    cmap = plt.cm.get_cmap('Oranges',cbar_nr), 
                    vmin=cbar_min, vmax=cbar_max, extend='both')
ax0.set_extent([300,348,45,82])
# Set colorbar
cbar = fig0.colorbar( sub0, ticks=cbar_tick )
cbar.ax.tick_params(labelsize=16)
cbar.set_label(cbar_label, fontsize=20) 

# Set boxes
for i in range(len(lon_bnd)):
    l1 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                  lat_bnd[i,0]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                  c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
    l2 = plt.plot(lon_bnd[i,0]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                  np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                  c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
    l3 = plt.plot(np.arange(lon_bnd[i,0], lon_bnd[i,1]+1,1), 
                  lat_bnd[i,1]*np.ones(lon_bnd[i,1]-lon_bnd[i,0]+1), 
                  c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())
    l4 = plt.plot(lon_bnd[i,1]*np.ones(lat_bnd[i,1]-lat_bnd[i,0]+1), 
                  np.arange(lat_bnd[i,0],lat_bnd[i,1]+1,1), 
                  c=linc, lw=linw, ls=lins[i], transform=ccrs.PlateCarree())

# Save if ordered
if saveplot == True:
    plt.savefig(dir_save+"/figs/"+var+"_"+per+"_count_mld1000_"\
                +repr(mld_bound)+"_poster.pdf")
plt.show()