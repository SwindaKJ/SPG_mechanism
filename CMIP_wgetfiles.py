#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:43:56 2022

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

Code to download wgetfiles of CMIP data. These are used to download the actual
data in the index computation without flooding the memory with all data.

"""

#%% Import modules and functions

import os
import numpy as np

# esgf 
from pyesgf.search import SearchConnection


#%% LIST OF SERVER NODES

node_list = ['https://esgf-node.llnl.gov/esg-search',
             'https://esgf-node.ipsl.upmc.fr/esg-search',
             'https://esgf-data.dkrz.de/esg-search',
             'https://esgf.ceda.ac.uk/esg-search']

#%% FUNCTIONS

def models_avail(conn, var_list=['msftmz','msftyz'],
                 exp_id='piControl', freq='mon', 
                 facets='project,experiment_id,source_id,variable,grid_label'):
    """
    Get a list of CMIP models that are available for all variables for a given 
    experiment and frequency.
    
    Default are the two streamfunction variables:
        - msftmz: Ocean Meridional Overturning Mass Streamfunction (AMOC)
        - msftyz: Ocean Y Overturning Mass Streamfunction (AMOC)
        >>> NOTE:   They will be available for different models, but are the
                    same for a lon-lat grid:
                    https://github.com/ESMValGroup/ESMValCore/issues/310
    for the piControl runs and a monthly frequency.

    Parameters
    ----------
    conn : Connection to one of the esgf nodes.
    var_list : List of variables. The default is ['msftmz','msftyz'].
    exp_id : Experiment ID. The default is 'piControl'.
    freq : Frequency. The default is 'mon'.
    facets : Facets for the search contaxt. The default is 'project, 
             experiment_id, source_id, variable, grid_label'.

    Returns
    -------
    source_list : A list of all models for which the variables are available.

    """
    
    # Set counter
    count = 0
    # Go through the list of variables
    for var in var_list:
        # Create a context for the variable
        ctx = conn.new_context(project='CMIP6', facets=facets,
                               experiment_id=exp_id, frequency=freq,
                               variable=var)
        # Get the available models and how many results
        source_avail = ctx.facet_counts['source_id']
        
        # Select models which do have data for each of the relevant variables
        # For the streamfunction one of the two variables is sufficient
        if count==0: # For the first variable, set initial output
            source_list = list(source_avail.keys())
            
        elif count!=0 and var in ['msftmz','msftyz']: # If streamfunction var
            [source_list.append(sc) for sc in source_avail.keys() 
             if sc not in source_list] # Append
            
        else: # Check which models are found in both lists
            source_temp = np.copy(source_list)
            source_list = [sc for sc in source_avail.keys() 
                           if sc in source_temp]
        
        # Add one
        count+=1
    
    # Sort alphabetically
    source_list = sorted(source_list)
    
    return source_list


def download_areawget(conn, dir_path, mod,
                      facets='project,experiment_id,source_id,variable,grid_label'):
    """
    Save the area files for a given model. If no file is available, save one
    for a model which (I expect) has the same grid.

    Parameters
    ----------
    conn : Connection to one of the esgf nodes.
    dir_path : The main directory in which to save the files.
    mod : The model.
    facets : Facets for the search contaxt. The default is 'project, 
             experiment_id, source_id, variable, grid_label'.

    Returns
    -------
    bool

    """
    # Set list of experiment id's to consider
    expid_area = ['piControl', 'historical', 'ssp585-bgc', 'other']
    
    # For both area variables
    for var in ['areacello']: #['areacello','areacella', 'volcello']:
        
        # Set directory for each variable
        vardir_path = os.path.join( dir_path, var )
        # If the directory does not exist, create it
        if not os.path.exists(vardir_path):
            os.makedirs(vardir_path)
        
        # Create context
        ctx = conn.new_context(project='CMIP6',facets=facets,
                               experiment_id=None, variable=var, source_id=mod)
        # Constrain to one experiment
        for expid in expid_area:
            # Check for which experiment area file available
            if expid in ctx.facet_counts['experiment_id']:
                # Constrain the context
                ctx = ctx.constrain(experiment_id = expid)
                break
            # If unavailable
            elif expid == 'other':
                # If UKESM1-1-LL, get from UKESM1-0-LL
                if mod == 'UKESM1-1-LL':
                    ctx = conn.new_context(project='CMIP6',facets=facets,
                                           experiment_id='piControl', 
                                           variable=var, 
                                           source_id='UKESM1-0-LL')
                # If CIESM, get from CESM
                elif var == 'areacella' and mod == 'CIESM':
                    ctx = conn.new_context(project='CMIP6',facets=facets,
                                           experiment_id='piControl', 
                                           variable=var,
                                           source_id='CESM2')
                # If EC-Earth-LR, get from EC-Earth-Veg-LR
                elif var == 'areacella' and mod == 'EC-Earth3-LR':
                    ctx = conn.new_context(project='CMIP6',facets=facets,
                                           experiment_id='historical', 
                                           variable=var, 
                                           source_id='EC-Earth3-Veg-LR')
                    
        # If MPI-ESM-1-2-HAM, get from MPI-ESM1-2-LR (unexecutable file, 
        # so as final correction)
        if mod == 'MPI-ESM-1-2-HAM':
            ctx = conn.new_context(project='CMIP6',facets=facets,
                                   experiment_id='piControl', variable=var, 
                                   source_id='MPI-ESM1-2-LR')
        # If INM-CM5-0, get from INM-CM4-8 (unexecutable file, so as final 
        # correction)
        elif var == 'areacella' and mod == 'INM-CM5-0':
            ctx = conn.new_context(project='CMIP6',facets=facets,
                                   experiment_id='piControl', variable=var, 
                                   source_id='INM-CM4-8')
        # If EC-Earth-LR, get from EC-Earth-Veg-LR
        elif var == 'areacello' and mod == 'EC-Earth3':
            ctx = conn.new_context(project='CMIP6',facets=facets,
                                   experiment_id='piControl', variable=var, 
                                   source_id='EC-Earth3-Veg')
        # If GISS-E2-1-G-CC, get from GISS-E2-1-G
        elif mod == 'GISS-E2-1-G-CC':
            ctx = conn.new_context(project='CMIP6',facets=facets,
                                   experiment_id='piControl', variable=var, 
                                   source_id='GISS-E2-1-G')
    
        # Get files
        results = ctx.search(facets=facets)
    
        # Get list of the available grids
        grid_avail = []
        for res in results:
            # Get the file info
            file_ctx = res.file_context()
            file_items = list(file_ctx.facet_constraints.items())[0][1].split(".")
            # Get list of grid
            if file_items[8] not in grid_avail:
                grid_avail.append(file_items[8])
        
        # For each of the results
        for res in results:
            
            # Get the file info
            file_ctx = res.file_context()
            file_items = list(file_ctx.facet_constraints.items())[0][1].split(".")
            # Rename to model we want to use area file for
            file_items[3] = mod
            print(file_items)
            
            # For grid we ideally have gn
            if file_items[8] in ['gr', 'gr1', 'gr2'] and 'gn' in grid_avail:
                print("Grid: " + file_items[8])
            
            else:
                # Set file name
                ind = [0, 3, 4, 5, 6, 7, 8]
                file_name = [file_items[i] for i in ind]
                file_name = ".".join(file_name) + ".sh"
            
                # Download wget files
                wget_script_content = file_ctx.get_download_script(facets=
                                                                    facets)
                # Set where to save
                script_path = os.path.join(vardir_path, file_name)
                # Create file to save the wget script as .sh executable
                with open(script_path, "w") as writer:
                    writer.write(wget_script_content)
                # Make wget script executable
                os.chmod(script_path, 0o750)
                print("Downloaded wget script: " + file_name)
                
                # Check whether the script is executable
                firstline = open(script_path, 'r').readline()
                # If not starts with shebang
                if not firstline.startswith('#!'):
                    os.remove(script_path)
                    print("Not executable: " + file_name)
                # Don't look for next model variant
                break
    
    return True


def download_wget(conn, dir_path, mod, var, exp_id='piControl', freq=None,
                  facets='project,experiment_id,source_id,variable,grid_label'):
    """
    Download the wgetfile for a given model, variable, experiment and variable.
    Checks for the desired grid (gn) and frequency (Amon or Omon for monthly)
    are included.

    Parameters
    ----------
    conn : Connection to one of the esgf nodes.
    dir_path : The main directory in which to save the files.
    mod : The model.
    var : The variable.
    exp_id : The experiment id. The default is 'piControl'.
    freq : The frequency of the data (monthly mainly. The default is None.
    facets : Facets for the search contaxt. The default is 'project, 
             experiment_id, source_id, variable, grid_label'.

    Returns
    -------
    bool

    """
    # Set directory for each variable
    vardir_path = os.path.join( dir_path, var, exp_id )
    # If the directory does not exist, create it
    if not os.path.exists(vardir_path):
        os.makedirs(vardir_path)
    
    # Create context
    ctx = conn.new_context(project='CMIP6', facets=facets,
                           experiment_id=exp_id, frequency=freq,
                           variable=var, source_id=mod)
    print(ctx.hit_count)
    
    # Get files
    results = ctx.search(facets=facets)
    
    # Get list of the available grids
    grid_avail = []
    for res in results:
        # Get the file info
        file_ctx = res.file_context()
        file_items = list(file_ctx.facet_constraints.items())[0][1].split(".")
        # Get list of grid
        if file_items[8] not in grid_avail:
            grid_avail.append(file_items[8])
    
    # Initialise lists of model variants and available frequencies
    mod_variants = [];      freq_avail = []
    # For each of the results
    for res in results:
        
        # Get the file info
        file_ctx = res.file_context()
        file_items = list(file_ctx.facet_constraints.items())[0][1].split(".")
        print(file_items)
        
        # Get list of frequencies
        if file_items[6] not in freq_avail:
            freq_avail.append(file_items[6])
        
        # For grid we ideally have gn
        if file_items[8] in ['gr', 'gr1', 'gr2'] and 'gn' in grid_avail:
            print("Grid: " + file_items[8])
        
        # So we look at the next result
        else:
            # For a new model variant
            if file_items[5] not in mod_variants:
                # Get list of model variants
                mod_variants.append(file_items[5])
                print(mod_variants)
                
                # For monthly frequencies we only want Amon and Omon, skip if not
                # AERmon???
                if freq=='mon' and not file_items[6] in ['Amon', 'Omon', 'SImon']: #, 'AERmon']:
                    mod_variants.pop()
                    print("Frequency: " + file_items[6])
                    
                # If all ok
                else:
                    # Set file name
                    ind = [0, 3, 4, 5, 6, 7, 8]
                    file_name = [file_items[i] for i in ind]
                    file_name = ".".join(file_name) + ".sh"
                
                    # Download wget files
                    wget_script_content = file_ctx.get_download_script(facets=
                                                                        facets)
                    
                    # Set where to save
                    script_path = os.path.join(vardir_path, file_name)
                    # Create file to save the wget script as .sh executable
                    with open(script_path, "w") as writer:
                        writer.write(wget_script_content)
                    # Make wget script executable
                    os.chmod(script_path, 0o750)
                    print("Downloaded wget script: " + file_name)
                    
                    # Check whether the script is executable
                    firstline = open(script_path, 'r').readline()
                    # If not starts with shebang
                    if not firstline.startswith('#!'):
                        os.remove(script_path)
                        mod_variants.pop()
                        print("Not executable: " + file_name)
    
    return True


#%% AREA DOWNLOAD

# Get connection
conn = SearchConnection(node_list[0], distrib=True)

# Get list of available models for a selected SPG variable
source_list = models_avail(conn, var_list=['thetao'])

# Set directory and facets
dir_path = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links/CMIP6_wgetfiles/'
facets = 'project, experiment_id, source_id, variable, grid_label'

# Download the area files for each model
for mod in source_list:
    print(mod)
    download_areares = download_areawget(conn, dir_path, mod)


#%% VARIABLE DOWNLOAD

# List of variables to download
var_list = ['msftbarot', 'tos', 'sos', 'mlotst', 'thetao', 'so']

# Set experiment id and frequency
exp_id = 'piControl'
freq = 'mon'

# Get connection
conn = SearchConnection(node_list[0], distrib=True)

# Download the variables for each model
for mod in source_list: 
    print(mod)
    # For each variable
    for var in var_list:
        print(var)
        download_res = download_wget(conn, dir_path, mod, var, exp_id, freq, 
                                     facets)

    




    


