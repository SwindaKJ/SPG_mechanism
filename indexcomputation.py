#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:35:39 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)


Index computation.

"""
#%% IMPORT MODULES

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

import os
import numpy as np
import xarray as xr
import random

from netCDF4 import Dataset
    

#%% SET DIRECTORIES

# Lorenz
if os.getcwd()[0:9] == '/storage/':
    # Main directory where wget files are stored
    dir_path = '/storage/home/3753808/Code/CMIP6_wgetfiles/'
    # Saving directory
    savedir_path = '/storage/home/3753808/Code/CMIP6_indices/'
    # Move to the directory where the function files are stored
    os.chdir('/storage/home/3753808/Code/')
    
    
# Local system
elif os.getcwd()[0:7] == '/Users/':
    # Main directory where wget files are stored
    dir_path = '/Users/3753808/Library/CloudStorage/' \
                'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                'CMIP6_wgetfiles/'
    # Saving directory
    savedir_path = '/Users/3753808/Library/CloudStorage/' \
                'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                'CMIP6_indices/'
    # Move to the directory where the function files are stored
    os.chdir('/Users/3753808/Library/CloudStorage/'
             'OneDrive-UniversiteitUtrecht/Code/Tipping_links/')
    
else:
    print("Where the heck are you running? Transfer to the right directory.")
    print(os.getcwd())


#%% IMPORT FUNCTIONS 

from indexcomputationfunctions.indexdatafunctions import login, \
    connect_to_server, obtain_wgetlist, download_data, wrapper, \
    download_areavar
import indexcomputationfunctions.indexfunctions as indfunc


#%% INPUT

index_list = ["amoc_m", "amoc_y", "sstatlspg", "sssatlspg", "mldatlspg", 
              "maxsfspg", "avsfspg", "varsfspg", "subthspg", "subsospg", 
              "subrhospg", "sshatlspg", "rhoatlspg"]

# Lorenz, give index as input
if os.getcwd()[0:9] == '/storage/':
    indexname = sys.argv[1]
    dom = sys.argv[2]
    lev0 = sys.argv[3]
    lev1 = sys.argv[4]
    startmod = int(sys.argv[5])
    
    lev_bnd = [int(lev0), int(lev1)]

# Local, set index name manually
elif os.getcwd()[0:7] == '/Users/':
    print("Set the index manually. For testing.")
    indexname = "subrhospg" #"subthspg"
    dom = 'std'
    lev_bnd = [50,1000]
    startmod = 0
    
else:
    print("Get yourself together. You are amazing.")


#%% FUNCTIONS

def index_settings(indexname, dom='std', lev_bnd=[50,1000]):
    """
    For the index of interest, get the corresponding variable and index
    computation function.

    Parameters
    ----------
    indexname : The name of the index: amoc_m, amoc_y, nino12, nino3, nino34, 
        nino4, twpaceast, twpaccentral, twpacwest, twatl, sstatlnorth, 
        sstatlequator, sstatlsouth, sstatlfingerprint.
    dom : The domain to be considered.
    lev_bnd : The vertical level bounds.

    Returns
    -------
    var : The variable name.
    index_func : Function. A function to compute the index of interest from the
        relevant variable.

    """

    # Set the variable that corresponds to the index:
    if indexname == "amoc_m":
        var = 'msftmz'
        def index_func(var, area, lev_bnd, dom):
            return indfunc.amoc_index(var)
    elif indexname == "amoc_y":
        var = 'msftyz'
        def index_func(var, area, lev_bnd, dom):
            return indfunc.amoc_index(var)

    # All sst variables
    elif indexname in ["sstatlspg"]:
        var = 'tos'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
        
    # All ocean temperature at depth variables
    elif indexname in ["subthspg"]:
        var = 'thetao'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, lev_bnd, dom)
    
    # All sss variables
    elif indexname in ["sssatlspg"]:
        var = 'sos'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
    
    # When we want density variables
    elif indexname in ["rhoatlspg"]:
        var = ['tos','sos']
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
    
    # When we want density at depth
    elif indexname in ["subrhospg"]:
        var = ['thetao','so']
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
    
    # All salinity at depth variables
    elif indexname in ["subsospg"]:
        var = 'so'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, lev_bnd, dom)
    
    # All sss variables
    elif indexname in ["sshatlspg"]:
        var = 'zos'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
    
    # Barotropic streamfunction
    elif indexname in ["maxsfspg", "avsfspg", "varsfspg"]:
        var = 'msftbarot'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)
        
    # Mixed layer depth
    elif indexname in ["mldatlspg"]:
        var = 'mlotst'
        def index_func(var, area, lev_bnd, dom):
            return getattr(indfunc, indexname+"_index")(var, area, dom)

    return var, index_func


def files_to_index(var, index_func, dom, lev_bnd, file_list, download_dir, dir_path):
    """
    Compute the index of interest. Load the required variables, select the 
    required regions and compute the anomaly when needed. Delete the files when
    finished.

    Parameters
    ----------
    var : The variable required for the index computation.
    index_func
    dom : The domain to be considered.
    lev_bnd : The vertical level bounds.
    file_list : A list of all files (different time sections) for one model of 
        the variable.
    download_dir : The directory in which the data is stored.
    dir_path : The path to the overarching directory.

    Returns
    -------
    index : A timeseries of the desired index.

    """
    # Print number of files
    print(len(file_list[0]))
    
    # Initialize counter
    count = 0
    
    # HadGEM3-GC31-MM crashes the kernel, so subdivide
    if file_list[0][0].split('_')[2] =='HadGEM3-GC31-MM' and 'thetao' in var:
        print('HadGEM3-GC31-MM')
        # For every file
        for i in range(len(file_list[0])):
            print(i)
            
            # Initialise lists for variable and area DataArrays
            var_cs = [[] for k in range(len(download_dir))]
            
            # For each variable
            for k in range(len(download_dir)):  
                print("Load data")
                # Create path to file
                filepath = os.path.join(download_dir[k], file_list[k][i])
                # Load dataset
                var_dataset = Dataset(filepath, mode='r')
                # Transfer to xarray
                var_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(var_dataset))
            
                print("Subdivide HadGEM3-GC31-MM data")
                for t in range(int(np.ceil(len(var_xr.time)/12))):
                    # Split by one year for wrapper
                    var_xr_p = var_xr.isel(time=slice(12*t,12*t+12))
                    var_cs_p = wrapper(var_xr_p)
                    
                    print("Need area file")
                    # Download corresponding cell area (only once per model)
                    if count==0 and not 'lev' in var_cs_p.lat.coords and k==0:
                        # Somehow GISS tos and sos data is stored on atmosphere grid
                        if var_cs_p.attrs['parent_source_id'][0:4] == 'GISS' \
                            and var in ['tos', 'sos']:
                            area_cs, area_filepath = download_areavar(
                                indexname, file_list[k][i].split('_')[5],
                                'areacella', var_cs_p.attrs['parent_source_id'], 
                                dir_path)
                            area_cs = area_cs.rename(areacella = 'areacello')
                        else:
                            area_cs, area_filepath = download_areavar(
                                indexname, file_list[k][i].split('_')[5],
                                var_cs_p.attrs['external_variables'].split(' ')[0],
                                var_cs_p.attrs['parent_source_id'], dir_path)
                        print("Downloaded area.")
                    elif 'lev' in var_cs_p.lat.coords:
                        print("Grid changes with depth. PANIC!")
                        # return "Failed"
                        
                    # Compute the index for one file
                    index0 = index_func(var_cs_p, area_cs, lev_bnd, dom)
                    print(index0)
                    print("Computed index.")

                    # Append in time over all files
                    if count == 0: # Create for first file
                        index_all = index0.copy()
                    else: # Append the others
                        index_all = xr.concat([index_all, index0.copy()], dim="time")
                        
                    print(index_all)
                    # Add one to count
                    count+=1
    
    else:
        # For every file
        for i in range(len(file_list[0])):
            print(i)
            
            # Initialise lists for variable and area DataArrays
            var_cs = [[] for k in range(len(download_dir))]
            
            # For each variable
            for k in range(len(download_dir)):  
                print("Load data")
                # Create path to file
                filepath = os.path.join(download_dir[k], file_list[k][i])
                # Load dataset
                var_dataset = Dataset(filepath, mode='r')
                # Transfer to xarray
                var_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(var_dataset))
                        
                # print(var_xr)
        
                if var in ['msftmz', 'msftyz']: # No area or wrapper needed for amoc
                    var_cs[k] = var_xr.copy()
                    area_cs = None
                else:
                    print("Preprocessing")
                    # if var_xr.attrs['parent_source_id'] in ["CESM2"]:
                    if len(var_xr.time) > 800:
                        # or var_xr.attrs['parent_source_id'] == 'HadGEM3-GC31-MM':
                        print("Separate in time before wrapper.")
                        # Loop over time in steps of 10
                        for t in range(int(np.ceil(len(var_xr.time)/12))):
                            # Select time frame
                            if t < len(var_xr.time)/12 - 1: # All steps
                                var_t = var_xr.isel(time=slice(12*t,12*(t+1)))
                            else: # Except the final one in case it is shorter
                                var_t = var_xr.isel(time=slice(12*t,len(var_xr.time)))
                            print(t)
                         
                            if t == 0: # First step
                                var_cs[k] = wrapper(var_t)
                            else: # Append
                                var_sofar = var_cs[k].copy()
                                var_pnew =  wrapper(var_t)
                                var_cs[k] = xr.concat([var_sofar,var_pnew], dim='time')
                    else:
                        var_cs[k] = wrapper(var_xr)
                    print("Need area file")
                    # Download corresponding cell area (only once per model)
                    if count==0 and not 'lev' in var_cs[0].lat.coords and k==0:
                        # Somehow GISS tos and sos data is stored on atmosphere grid
                        if var_cs[0].attrs['parent_source_id'][0:4] == 'GISS' \
                            and var in ['tos', 'sos', ['tos', 'sos']]:
                            print("GISS")
                            area_cs, area_filepath = download_areavar(
                                indexname, file_list[k][i].split('_')[5],
                                'areacella', 'GISS-E2-1-G', 
                                dir_path)
                            area_cs = area_cs.rename(areacella = 'areacello')
                        else:
                            area_cs, area_filepath = download_areavar(
                                indexname, file_list[k][i].split('_')[5],
                                var_cs[0].attrs['external_variables'].split(' ')[0],
                                var_cs[0].attrs['parent_source_id'], dir_path)
                        print("Downloaded area.")
                    elif 'lev' in var_cs[k].lat.coords:
                        print("Grid changes with depth. PANIC!")
                        # return "Failed"
            print(var_cs)
            
            # If just one variable, get rid of the list part (ensures earlier stuff
            # still runs)
            if isinstance(var, str):
                # For MRI and MIROC6, remove dimensions
                if var_cs[0].attrs['parent_source_id'] in ['MRI-ESM2-0','MIROC6']:
                    print("Drop vertex and bnds dimensions")
                    if count==0:
                        area_cs = area_cs.drop_dims(['vertex','bnds'])
                    var_cs[0] = var_cs[0].drop_dims(['vertex','bnds'])
            
            # Compute the index for one file
            print("Start index computation.")
            if len(var_cs[0].time) > 800:
                # or var_xr.attrs['parent_source_id'] == 'HadGEM3-GC31-MM':
                print("Separate in time before index computation.")
                if isinstance(var, str):
                    var_cs = var_cs[0]
                # Loop over time in steps of 10
                for t in range(int(np.ceil(len(var_cs.time)/12))):
                    # Select time frame
                    if t < len(var_cs.time)/12 - 1: # All steps
                        var_t = var_cs.isel(time=slice(12*t,12*(t+1)))
                    else: # Except the final one in case it is shorter
                        var_t = var_cs.isel(time=slice(12*t,len(var_cs.time)))
                    
                    # Compute index for each subset
                    if t == 0: # First step
                        index0 = index_func(var_t, area_cs, lev_bnd, dom)
                    else: # Append
                        index0_sofar = index0.copy()
                        index0_pnew =  index_func(var_t, area_cs, lev_bnd, dom)
                        index0 = xr.concat([index0_sofar,index0_pnew], dim='time')
            else:
                if isinstance(var, str):
                    var_cs = var_cs[0]
                index0 = index_func(var_cs, area_cs, lev_bnd, dom)
            print(index0)
            print("Computed index.")
    
            # Append in time over all files
            if count == 0: # Create for first file
                index_all = index0.copy()
            else: # Append the others
                index_all = xr.concat([index_all, index0.copy()], dim="time")
                
            print(index_all)
            # Add one to count
            count+=1

    # Anomaly computation
    print("Compute anomaly")
    if var in ['msftmz', 'msftyz', 'mlotst']: # No anomalies for amoc or MLD
        index = index_all.copy()
    else: # Compute anomalies for all other indices
        index = indfunc.index_anomaly(index_all)

    return index


def save_index(indexname, dom, lev_bnd, index, savedir_path, wget_name):
    """
    Save the computed index variable.

    Parameters
    ----------
    indexname : Name of the index.
    dom : The domain for which the index has been computed.
    lev_bnd : The vertical level bounds.
    index : Timeseries of the index.
    savedir_path : Path to the directory where to save the file.
    wget_name : The wget script corresponding to the model, to use as basis for
        the name under which to save the index timeseries.

    Returns
    -------
    None.

    """

    # Save results
    save_dir = os.path.join(savedir_path, indexname, dom, 'piControl')
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Get the name for saving
    if not indexname[0:3] == 'sub': # If 2D data
        save_name = wget_name.replace('.sh','.'+indexname+'.'+dom+'.nc')
    else: # If 3D data
        save_name = wget_name.replace('.sh','.'+indexname+'.'+dom+'.lev'
                                      +repr(lev_bnd[0])+'-'+repr(lev_bnd[1])+'.nc')
    # Save as netcdf
    index.to_netcdf(os.path.join(save_dir, save_name))
    return


def check_indexcomputed(indexname, dom, lev_bnd, savedir_path, mod_version):
    """
    A function to check whether an index has already been computed for a given
    model (wgetfile).

    Parameters
    ----------
    indexname : Name of the index.
    dom : The domain for which the index has been computed.
    lev_bnd : The vertical level bounds.
    savedir_path : Path to the directory where to save the file.
    mod_version: The model and version.

    Returns
    -------
    True is the index-file already exists, False if it does not.

    """
    
    # Get directory where everything is saved
    save_dir = os.path.join(savedir_path, indexname, dom, 'piControl')
    # Get the name for saving
    save_name_start = 'CMIP6.'+mod_version.split('_')[0]+'.piControl.'\
                        +mod_version.split('_')[1]
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # If the index already has been computed, return True
    if len([x for x in os.listdir(save_dir) if 
            x.startswith(save_name_start)]) > 0:
        print("The index has already been computed for this model.")
        return True
    else:
        return False


def compute_index(indexname, dom, startmodel, dir_path, savedir_path):

    # Get the variable and index computation function 
    var, index_func = index_settings(indexname, dom, lev_bnd)
    print(var)

    if isinstance(var, str): # If just one variable
        print("One variable")
        # Get a list of all wgetfiles for the given variable and experiment
        wget_list, vardir_path = obtain_wgetlist(var, 'piControl', dir_path)
        # Get a list of all models and versions
        mod_version_avail = [0]*len(wget_list)
        for j in range(len(wget_list)):
            mod_version_avail[j] = wget_list[j].split('.')[1]+"_"\
                                    +wget_list[j].split('.')[3]
        mod_version_list = mod_version_avail.copy()
    elif isinstance(var, list): # If multiple variables
        print("Multiple variables")
        # Initialise
        wget_list = [[] for i in range(len(var))]
        vardir_path = [[] for i in range(len(var))]
        # Get a list of all wgetfiles for the each of the variables and given experiment
        for i in range(len(var)):
            wget_list[i], vardir_path[i] = obtain_wgetlist(var[i], 'piControl', 
                                                           dir_path)
        # Get a list of all models and versions
        mod_version_list = [[0]*len(wget_list[i]) for i in range(len(var))]
        for i in range(len(var)):
            for j in range(len(wget_list[i])):
                mod_version_list[i][j] = wget_list[i][j].split('.')[1]+"_"\
                                        +wget_list[i][j].split('.')[3]
        # Get overlap between lists
        mod_version_avail = list(set.intersection(*map(set,mod_version_list))) 
    else:
        print("Something went wrong. Check your variable.")
    
    # Sort in sensible order
    mod_version_avail.sort()
    
    # For every wget file
    for mod_version in mod_version_avail[startmodel::]:#[37:39]:#[startmodel::]:
        # Print the model (version) name
        print(mod_version)
        
        # If the index has already been computed, stop
        if check_indexcomputed(indexname, dom, lev_bnd, savedir_path, 
                                mod_version) == True:
            # Skip to the next one
            continue
        
        # # Check whether it's the right model version
        if mod_version.split('_')[1] not in ['r1i1p1f1','r1i1p1f2']:
            print("Model version: "+ mod_version.split('_')[1])
            continue
        
        # Random server
        server_nr = random.randint(0,3)
        # Connect to a server
        conn = connect_to_server(server_nr)
        print("Server Nr: "+repr(server_nr))
        
        # If the index has already been computed, stop
        if check_indexcomputed(indexname, dom, lev_bnd, savedir_path, 
                                mod_version) == True:
            # Skip to the next one
            continue

        try:
            # Download data
            print("Download data")
            if isinstance(var, str): # If just one variable
                print("Download try: " + repr(0))
                download_dir = [[]];    file_list = [[]]
                # Get corresponding wget file
                if mod_version[0:6] == 'NorESM' and var == 'thetao': # Get z-grid
                    wget_name = np.array(wget_list)\
                        [np.array(mod_version_list)==mod_version][1]
                else:
                    wget_name = np.array(wget_list)\
                        [np.array(mod_version_list)==mod_version][0]
                # Download data
                download_dir[0], file_list[0] = \
                    download_data(indexname, wget_name, vardir_path)
                    
                # Check for 0 byte files
                file_size = [[] for i in range(len(file_list[0]))]
                for i in range(len(file_list[0])):
                    file_size[i] = os.stat(download_dir[0]+'/'
                                           +file_list[0][i]).st_size
                # Initialize, while at least one 0 byte file
                dwnld_nr = 1
                while 0 in file_size and dwnld_nr <= 10:
                    print("Number of zero byte files: " + 
                          repr(np.count_nonzero(np.array(file_size) == 0)))
                    print("Download try: " + repr(dwnld_nr))
                    
                    # Random server
                    server_nr = random.randint(0,3)
                    # Connect to a server
                    conn = connect_to_server(server_nr)
                    print("Server Nr: "+repr(server_nr))
                    
                    # Create new directory for downloading data again
                    download_dir_new = os.path.join(download_dir[0],"Try")
                    if not os.path.exists(download_dir_new):
                        os.makedirs(download_dir_new)
                    # Check whether directory is empty
                    if not len(os.listdir(download_dir_new)) == 0:
                        print("Folder not empty.")
                        # Empty directory
                        for file in os.listdir(download_dir_new):
                            if file[0] != '.':
                                os.remove(os.path.join(download_dir_new, file))
                    
                    # Download data again in new directory
                    download_dir_new, file_list_new = \
                        download_data(indexname, wget_name, vardir_path, 
                                      download_dir_new)
                    
                    # Get list of new file sizes and replace zero byte files 
                    # with new ones if new file is non-zero
                    for i in range(len(file_list[0])):
                        file_size_new = os.stat(download_dir_new+'/'
                                                +file_list_new[i]).st_size
                        if file_size[i] == 0 and file_size_new != 0:
                            os.replace(download_dir_new+'/'+file_list_new[i],
                                       download_dir[0]+'/'+file_list[0][i])
                    
                    # Update file list (Check)
                    file_list_all = os.listdir(download_dir[0])
                    file_list_all.sort()
                    file_list[0] = [file for file in file_list_all 
                                    if not file.startswith('.') 
                                    and not file == 'Try']
                    # and list of file sizes
                    for i in range(len(file_list[0])):
                        file_size[i] = os.stat(download_dir[0]+'/'
                                               +file_list[0][i]).st_size
                    dwnld_nr += 1
                
            elif isinstance(var, list): # If multiple variables
                # Initialise
                download_dir = [[] for i in range(len(var))]
                file_list = [[] for i in range(len(var))]
                for i in range(len(var)): # For each variable
                    # Get corresponding wget file
                    wget_name = np.array(wget_list[i])\
                        [np.array(mod_version_list[i])==mod_version][0]
                    # Download data
                    download_dir[i], file_list[i] = \
                        download_data(indexname, wget_name, vardir_path[i])
            
            # If more variables, check whether time of data overlaps
            if isinstance(var, list):
                print("Check overlap of datasets")
                filetime = [[] for k in range(len(file_list))]
                for k in range(len(file_list)):
                    filetime[k] = [file_list[k][i].split('_')[-1] 
                                   for i in range(len(file_list[k]))]
                # print(filetime) 
                # Get times that are available for all variables
                filetime_all = list(set.intersection(*map(set,filetime)))
                filetime_all.sort()
                # If time not available for both, delete from file_list
                for k in range(len(file_list)):
                    file_del = []
                    for i in range(len(file_list[k])):
                        if not filetime[k][i] in filetime_all:
                            file_del.append(file_list[k][i])
                    # print(file_del)
                    if len(file_del) > 0:
                        for j in range(len(file_del)):
                            file_list[k].remove(file_del[j])
                        print("Removed files")
                        print(file_del)
                print([len(file_list[k]) for k in range(len(file_list))])
            
            # If some overlap in files still
            if len(file_list[0]) > 0:
            
                # Compute the index for all files and put them together
                print("Compute index")
                index = files_to_index(var, index_func, dom, lev_bnd, 
                                       file_list, download_dir, dir_path)
                # Save the index
                print("Save index")
                save_index(indexname, dom, lev_bnd, index, savedir_path, 
                           wget_name)
            else:
                print("The different datasets are not alligned in time.")
                continue
    
        except Exception:
            print("Error")
            continue
        
    return


#%% COMPUTE THE INDEX

ci = compute_index(indexname, dom, startmod, dir_path, savedir_path)


#%% TESTING FOR 3D VARIABLES

# download_dir = '/Users/3753808/Library/CloudStorage/' \
#                 'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
#                 'SPG_data/ACCESS-ESM1-5/thetao/'
                
# # Get list of files
# file_list_all = os.listdir(download_dir)
# # Sort list
# file_list_all.sort()
# # Remove status
# file_list = [file for file in file_list_all if not file.startswith('.')]


#%%

# # Get the list of models and wgetfiles
# area_dir = '/Users/3753808/Library/CloudStorage/' \
#             'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
#             'CMIP6_wgetfiles/volcello/temp/subthspg/'

# # Get list of files
# area_list_all = os.listdir(area_dir)
# # Sort list
# area_list_all.sort()
# # Remove status
# area_list = [file for file in area_list_all if not file.startswith('.')]

# for i in range(1):
#     areaname = area_list[i]
#     # Create path to file
#     areapath = os.path.join(area_dir, areaname)
#     # Load dataset
#     area_dataset = Dataset(areapath, mode='r')
#     # Transfer to xarray
#     area_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(area_dataset))
    
#     print(area_xr)

#%% DEBUGGING

# import os, subprocess

# # Get the variable and index computation function 
# var, index_func = index_settings(indexname)
# print(index_func)

# # Get a list of all wgetfiles for the given variable and experiment
# wget_list, vardir_path = obtain_wgetlist(var, 'piControl', dir_path)

# For every wget file
# for wget_name in wget_list[startmod::]:
    # Print the model (version) name
#     print(wget_name)

#%%
# wget_name = wget_list[38]
# print(wget_name)

# # # Check whether it's the right model version
# if wget_name.split('.')[3] not in ['r1i1p1f1','r1i1p1f2']:
#     print("Model version: "+ wget_name.split('.')[3])
#     continue

# # If the index has already been computed, stop
# if check_indexcomputed(indexname, dom, lev_bnd, savedir_path, 
#                         wget_name) == True:
#     # Skip to the next one
#     continue
#%% MORE VARIABLES
"""
Get directory and list of files for downloads data with multiple variables. 
Ready for the index function.
"""
# wget_name = wget_list[30]
# print(wget_name)

# # Connect to a !server
# conn = connect_to_server(0)

# Download data
# Initialise
# download_dir = [[] for i in range(len(var))]
# file_list = [[] for i in range(len(var))]
# for i in range(len(var)): # For each variable
#     # Get corresponding wget file
#     wget_name = wget_list[i][mod_version_avail == 
#                               mod_version]
#     # Download data
#     # Set where to save
#     script_path = os.path.join(vardir_path[i], wget_name)
#     # Initialise temporary directory to download to
#     download_dir[i] = os.path.dirname(script_path) + "/temp/" + indexname 

#     # Get list of files
#     file_list_all = os.listdir(download_dir[i])
#     # Sort list
#     file_list_all.sort()
#     # Remove status
#     file_list[i] = [file for file in file_list_all if not file.startswith('.')]
# #%%
# # Check same number of files for each variable
# file_list_len = [len(file_list[i]) for i in range(len(file_list))]
# if all(ln == file_list_len[0] for ln in file_list_len):

#     # Compute the index for all files and put them together
#     print("Compute index")
#     index = files_to_index(var, index_func, dom, lev_bnd, 
#                            file_list, download_dir, dir_path)

# #%% TESTING DENSITY

# # Compute density anomaly
# rho_ar = gsw.density.rho_t_exact(var_cs[1].sos, var_cs[0].tos, 0) - 1000
# rho_cs = rho_ar.to_dataset().rename(sos='rho')
# print(rho_cs)

#%% ONE VARIABLE
"""
Get directory and list of files for downloads data with one variable. Ready 
for the index function.
"""
download_dir = [[]];    file_list = [[]]
# Get corresponding wget file
wget_name = np.array(wget_list)[np.array(mod_version_avail) == mod_version][0]
# Download data
# Set where to save
script_path = os.path.join(vardir_path, wget_name)
# Initialise temporary directory to download to
download_dir[0] = os.path.dirname(script_path) + "/temp/" + indexname 

# Get list of files
file_list_all = os.listdir(download_dir[0])
# Sort list
file_list_all.sort()
# Remove status
file_list[0] = [file for file in file_list_all if not file.startswith('.') and not file == 'Try']

#%% CHECK FILE SIZES

file_size = [[] for i in range(len(file_list[0]))]
for i in range(len(file_list[0])):
    file_size[i] = os.stat(download_dir[0]+'/'+file_list[0][i]).st_size

dwnld_nr = 1
while 0 in file_size and dwnld_nr <= 10:
    print("File of 0 bytes.")
    print("Download try: " + repr(dwnld_nr))
    # Create new directory for downloading data again
    download_dir_new = os.path.join(download_dir[0],"Try")
    if not os.path.exists(download_dir_new):
        os.makedirs(download_dir_new)
    
    # Download data again in new directory
    download_dir_new, file_list_new = \
        download_data(indexname, wget_name, vardir_path, download_dir_new)
    
    # Get list of new file sizes and replace zero byte files with new ones if 
    # new is non-zero
    file_size_new = [[] for i in range(len(file_list[0]))]
    for i in range(len(file_list[0])):
        file_size_new[i] = os.stat(download_dir_new+'/'+file_list_new[i]).st_size
        if file_size[i] == 0 and file_size_new != 0:
            os.replace(download_dir_new+'/'+file_list_new[i],
                       download_dir[0]+'/'+file_list[0][i])
    
    for i in range(len(file_list[0])):
        file_size[i] = os.stat(download_dir[0]+'/'+file_list[0][i]).st_size
    print("Number of zero byte files: " + 
          repr(np.count_nonzero(np.array(file_size) == 0)))
    dwnld_nr += 1

# MORGEN VERDER
        

#%%

# Get list of files
file_list_all = os.listdir(download_dir_new)
# Sort list
file_list_all.sort()
# Remove status
file_list_new = [file for file in file_list_all if not file.startswith('.')]


#%%
import random

lst = np.zeros(10)
cnt = 1

while 0 in lst and cnt <= 15:
    # print(cnt)
    lst[random.randint(0,9)] = 1
    # print("Number of zero byte files: " + 
    #       repr(np.count_nonzero(np.array(lst) == 0)))
    print(random.randint(0,3))
    cnt += 1
    


#%% AREA CHECK

# print(var_cs[0].attrs['external_variables'].split(' ')[0])
# print(var_cs[0].attrs['parent_source_id'])

# area_cs, area_filepath = download_areavar(
#     indexname, file_list[k][i].split('_')[5],
#     var_cs[0].attrs['external_variables'].split(' ')[0],
#     var_cs[0].attrs['parent_source_id'], dir_path)

#%%

# # Compute the index for all files and put them together
# print("Compute index")
# index = files_to_index(var, index_func, dom, lev_bnd, file_list, download_dir, 
#                         dir_path)

# # Save the index
# print("Save index")
# save_index(indexname, index, savedir_path, wget_name)


#%% Get CMCC right

# from xmip.preprocessing import rename_cmip6, promote_empty_dims, \
#     broadcast_lonlat, replace_x_y_nominal_lat_lon

# ds = var_xr.copy()
# ds = rename_cmip6(ds)
# ds = promote_empty_dims(ds)
# ds = broadcast_lonlat(ds)
# # ds = correct_lon(ds)
# ds = replace_x_y_nominal_lat_lon(ds)

# var_new = var_xr.rename(latitude="x", longitude="y")
# var_new = var_new.stack(ij=("i", "j"))
# var_new = var_new.set_index(ij=["x","y"])

#%%

# var = ['tos','sos','psl']

# # Initialise
# wget_list = [[] for i in range(len(var))]
# vardir_path = [[] for i in range(len(var))]
# # Get a list of all wgetfiles for the each of the variables and given experiment
# for i in range(len(var)):
#     wget_list[i], vardir_path[i] = obtain_wgetlist(var[i], 'piControl', 
#                                                    dir_path)
# # Get a list of all models and versions
# mod_version_list = [[0]*len(wget_list[i]) for i in range(len(var))]
# for i in range(len(var)):
#     for j in range(len(wget_list[i])):
#         mod_version_list[i][j] = wget_list[i][j].split('.')[1]+"_"\
#                                 +wget_list[i][j].split('.')[3]
# # Get overlap between lists
# mod_version_avail = list(set.intersection(*map(set,mod_version_list)))












