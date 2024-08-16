#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:33:03 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

Functions to download and preprocess CMIP data using wget files.

"""

#%% Import modules and functions

import os, subprocess
import cftime
import numpy as np
import xarray as xr

from netCDF4 import Dataset
from pyesgf.search import SearchConnection
import xmip.preprocessing as xmip_pre


#%% LOGIN

def connect_to_server(number):
    """
    Connect to one of esgf servers

    Parameters
    ----------
    number : The index for the node to connect to.

    Returns
    -------
    conn : Connection to one of the servers.

    """
    
    # List of all available nodes
    node_list = ['https://esgf-node.llnl.gov/esg-search',
                 'https://esgf-node.ipsl.upmc.fr/esg-search',
                 'https://esgf-data.dkrz.de/esg-search',
                 'https://esgf.ceda.ac.uk/esg-search']
    # Connect to server
    conn = SearchConnection(node_list[number], distrib=True)
    
    return conn

#%% FUNCTIONS

def download_data(indexname, file_name, dir_path, alt_download_dir =  None):
    """
    Download CMIP data using a given wgetfile, i.e. for one variable, model 
    and experiment id. The data is saved in a folder called 'temp' in the same 
    directory as where the wgetfile is located.

    Parameters
    ----------
    indexname : The name of the index.
    file_name : The wgetfile for the data to download.
    dir_path : Give the directory in which the wgetfile is located.
    alt_download_dir : Optional alternative directory to download data to.
        The default is None.

    Returns
    -------
    download_dir : The directory in which the data has been saved.
    file_list : The list of filenames of the downloaded data.

    """
    
    # Set where to save
    script_path = os.path.join(dir_path, file_name)
    # If alternative saving directory is given
    if alt_download_dir:
        download_dir = alt_download_dir
    else: # If not (standard)
        # Initialise temporary directory to download to
        download_dir = os.path.dirname(script_path) + "/temp/" + indexname 
        # If the directory does not exist, create it
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
    # Check whether directory is empty
    if not len(os.listdir(download_dir)) == 0:
        print("Folder not empty.")
        # Empty directory
        for file in os.listdir(download_dir):
            if file[0] != '.' and not file == "Try":
                os.remove(os.path.join(download_dir, file))

    # Download the data for the given model
    os.chmod(script_path, 0o750)
    subprocess.check_output(["{}".format(script_path), '-s'], cwd=download_dir)
    
    # Get list of files
    file_list_all = os.listdir(download_dir)
    # Sort list
    file_list_all.sort()
    # Remove status
    file_list = [file for file in file_list_all if not file.startswith('.')
                 and not file == 'Try']
        
    return download_dir, file_list


def obtain_wgetlist( var, exp_id, dir_path ):
    """
    Get a list of the wgetfiles for a given variable and experiment id. This 
    will (should) be a list with wgetfiles for each of the models for which
    the variable us available for the given experiment.
    >> Does rely on folder structure.

    Parameters
    ----------
    var : The variable name.
    exp_id : The experiment id.
    dir_path : The overarching directory in which the wgetfiles are located.

    Returns
    -------
    wget_list : A list of wgetfiles (for each model).
    vardir_path : The directory where the wgetfiles are located.

    """
    # Set path to the variable and experiment directory
    vardir_path = os.path.join( dir_path, var, exp_id )
    
    # List the files in the directory
    wget_list = os.listdir( vardir_path )
    wget_list.sort()
    
    # Select the wget files
    wget_list = [wget for wget in wget_list if wget.startswith('CMIP')]
    
    return wget_list, vardir_path

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

def correct_lon_180(ds): # FROM SJOERD
    """
    Wraps negative x and lon values around to have -180 - 180 lons.
    longitude names expected to be corrected with `rename_cmip6`.
    Adapted from xmip_pre.correct_lon()
    """
    ds = ds.copy()

    # remove out of bounds values found in some
    # models as missing values
    ds["lon"] = ds["lon"].where(abs(ds["lon"]) <= 1000)
    ds["lat"] = ds["lat"].where(abs(ds["lat"]) <= 1000)

    # adjust lon convention
    lon = ds["lon"].where(ds["lon"] <= 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=lon)

    if "lon_bounds" in ds.variables:
        lon_b = ds["lon_bounds"].where(ds["lon_bounds"] <= 180, ds["lon_bounds"] - 360)
        ds = ds.assign_coords(lon_bounds=lon_b)

    return ds

def wrapper(ds):
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
    if not ds.attrs['parent_source_id'] == 'IPSL-CM6A-LR':
        ds = xmip_pre.correct_units(ds)
    ds = xmip_pre.fix_metadata(ds)
    
    # GFDL model
    if "x_deg" in ds.coords:
        ds = ds.rename({'x_deg': 'x', 'y_deg':'y'})
    try:
        ds = ds.rename({"time_bounds": "time_bnds"})
    except ValueError:
        pass
    
    return ds


def datetime_to_cftime(dates, kwargs={}):
    """
    Convert the dates from a datetime format to cftime.

    Parameters
    ----------
    dates : A list of dates in datetime format.
    kwargs : Optional. The default is {}.

    Returns
    -------
    List of dates in cftime format.

    """
    return [ cftime.datetime(date.dt.year,
                              date.dt.month,
                              date.dt.day,
                              date.dt.hour,
                              date.dt.minute,
                              date.dt.second,
                              date.dt.microsecond,
                              **kwargs) 
            for date in dates]


def time_to_dayssince(xr):
    """
    Convert the time dimension of the CMIP DataArray to a 
    'days since 1850-01-01'-format for consistency.

    Parameters
    ----------
    xr : A CMIP DataArray with a dimension called 'time'.

    Returns
    -------
    xr : The same DataArray with the converted 'time' dimension.

    """
    # If not already converted
    if xr["time"].dtype not in ['float64']:
        # If the data is in datetime format, convert to cftime
        if xr["time"].dtype in ['<M8[ns]', 'datetime64[ns]']:
            print("Converted to cftime")
            xr["time"] = datetime_to_cftime(xr["time"])
        
        # Convert all times to 'days since' for consistency
        datenumber = cftime.date2num(xr.indexes["time"], 'days since 1850-01-01')
        xr["time"] = datenumber
    
    return xr


def arealist(dir_path, area_var):
    """
    For one of the CMIP area variables get a list of the available models, 
    the corresponding wgetfiles and the directory in which to find them.
    >> Does rely on folder structure.

    Parameters
    ----------
    dir_path : The overarching directory in which the wgetfiles are located.
    area_var : The area variable.

    Returns
    -------
    area_model : A list of all models for which the area variable is available.
    area_wgetlist : A list of the wgetfiles corresponding to each of the models.
    area_vardir_path : The directory in which the wgetfiles are located.

    """
    # Path to areacello or areacella variable
    area_vardir_path = os.path.join( dir_path, area_var )
    
    # List the files in the directory
    area_wgetlist = os.listdir( area_vardir_path )
    area_wgetlist.sort()
    
    # Select the wget files
    area_wgetlist = np.array([wget for wget in area_wgetlist 
                              if wget.startswith('CMIP')])
    
    # Get list of models corresponding to wget files
    area_model = np.array([mod.split(".")[1] for mod in area_wgetlist])
    
    return area_model, area_wgetlist, area_vardir_path 


def download_areavar(indexname, grid, area_var, mod, dir_path):
    """
    Download CMIP area files (areacello or areacella) for the model that is 
    considered using the corresponding wgetfile.

    Parameters
    ----------
    indexname : The index to look at.
    grid : The grid of the model.
    area_var : The area variable.
    mod : The model.
    dir_path : The directory in which the wgetfiles are located.

    Returns
    -------
    area_cs : The DataArray of the area data.
    area_filepath : The path to the location of the area data. ### NEEDED?

    """
    # Get the list of models and wgetfiles
    area_model, area_wgetlist, area_vardir_path = arealist(dir_path, area_var)
    
    # If the first part of the name is the same and correct grid
    for i in range(len(area_model)):
        if mod == area_model[i] and area_wgetlist[i].split('.')[6] == grid:
            print(i)
            # Download data
            area_download_dir, area_file_list = \
                download_data(indexname, area_wgetlist[i], area_vardir_path)
            # break
        elif mod == area_model[i] and mod[0:6] == "NorESM":
            print(i)
            # Download data
            area_download_dir, area_file_list = \
                download_data(indexname, area_wgetlist[i], area_vardir_path)
    
    # Create path to file
    area_filepath = os.path.join(area_download_dir, area_file_list[0])
    # Load dataset
    area_dataset = Dataset(area_filepath, mode='r')
    # Transfer to xarray
    area_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(area_dataset))
    
    print(area_xr)
    
    # Rename/change variables using XMIP for consistency
    area_cs = wrapper(area_xr)
    
    return area_cs, area_filepath