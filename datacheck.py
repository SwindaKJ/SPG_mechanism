#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:31:52 2023

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

To check the model index data for length, NaN, ...

"""

#%% IMPORT

import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import matplotlib
from matplotlib import pyplot as plt

#%% FUNCTIONS

def variable_selection(index_selection, dom, exp_id="piControl", 
                       dir_path="/Users/3753808/Library/CloudStorage/" \
                                "OneDrive-UniversiteitUtrecht/Code/" \
                                "Tipping_links/CMIP6_indices/"):
    """
    Select the variables to consider for the causal effect network.

    Parameters
    ----------
    index_selection : A list of the variable names.
    dom : The domain to be considered. The default is "std".
    exp_id : Optional. The experiment ID. The default is "piControl".
    dir_path : Optional. The path to where the data is stored. The default is 
        "/Users/3753808/Library/CloudStorage/OneDrive-UniversiteitUtrecht/Code/
        Tipping_links/CMIP6_indices/".

    Returns
    -------
    index_path : A list of the directory paths to the files in which each of 
        the variables is stored.
    index_list : A list of all the files in the variable directory (models).
    mod_list0 : A list of all available models for each of the variables.
    mod_list : A list of the models that are available for all variables.

    """
    # Initialise
    index_path = [[] for i in range(len(index_selection))]
    index_list = [[] for i in range(len(index_selection))]
    mod_list0 = [[] for i in range(len(index_selection))]
    # For each variable
    for i in range(len(index_selection)):
        # Set the path to the directory where the data is stored
        index_path[i] = os.path.join(dir_path, index_selection[i], dom[i], exp_id)
        # Get a list of files in that directory
        index_list[i] = [x for x in os.listdir(index_path[i]) if x[0] == 'C']
        index_list[i].sort()
        # Match the models (should be the same, but he)
        mod_list0[i] = np.array([filename.split('.')[1] + "_" 
                                 + filename.split('.')[3] 
                                 for filename in index_list[i]])
    
    # Select models which are available for all variables
    mod_list = np.copy(mod_list0[0])
    for i in range(1, len(index_selection)):
        mod_temp = np.array([mod for mod in mod_list0[i] if mod in mod_list])
        mod_list = np.copy(mod_temp)
    
    return index_path, index_list, mod_list0, mod_list


def data_load(mod, index_selection, index_path, index_list, mod_all_list):
    """
    Load the data for each model (all variables)

    Parameters
    ----------
    mod : The model.
    index_path : A list of the directory paths to the files in which each of 
        the variables is stored.
    index_list : A list of all the files in the variable directory (models).
    mod_all_list : A list of all available models for each of the variables.
    aggregation_time : The time over which to aggregate the data (years).
    time_data : A list of the types of data, annual, seasonal or monthly mean.
    time_ind : Optional. Indicate the month(s) to consider for seasonal and 
        monthly data The default is None.

    Returns
    -------
    data_var : A list containing the timeseries of each of the variables.

    """
    # Initialise
    data_xr = [[] for i in range(len(index_selection))]
    for i in range(len(index_selection)):
        # Select file names
        file = np.array(index_list[i])[mod_all_list[i]==mod][0]
        # Create path to file
        filepath = os.path.join(index_path[i], file)
        # Load dataset
        dataset = Dataset(filepath, mode='r')
        # Transfer to xarray
        data_xr[i] = xr.open_dataset(xr.backends.NetCDF4DataStore(dataset))
    
    return data_xr

def data_period(data_xr_list, index_sel, var_data, aggregation_time, time_data, 
                time_ind):
    
    # Get the length of the dataset for each variable
    len_data = [len(data_xr_list[i].time) for i in range(len(index_list))]
    print(len_data)
    
    # Ensure all variables are aligned in time
    data_sel = [[] for i in range(len(index_sel))]
    # If not all timeseries have the same length
    if not all(i == len_data[0] for i in len_data):
        # Create list of start and end dates
        start_list = [data_xr_list[i].time[0] for i in range(len(index_sel))]
        end_list = [data_xr_list[i].time[-1] for i in range(len(index_sel))]
        # Select the part corresponding to the latest start data and earliest 
        # end date
        for i in range(len(index_sel)):
            data_sel[i] = data_xr_list[i].sel(time = slice(np.amax(start_list), 
                                                           np.amin(end_list)))
    # Else just copy
    else:
        data_sel = data_xr_list
    
    print(data_sel)
    
    # Initialise data array for yearly, seasonal or monthly data
    data_var = np.zeros((len(index_sel), 
                         int(len(data_sel[0].time) /12 /aggregation_time)))
    # For each variable
    for i in range(len(index_sel)):
        # Depending on the time indication for that variable
        if time_data[i] == "year":
            # Moving average over 1 year
            year_series = pd.Series(np.array(data_sel[i][var_data[i]]))
            year_window = year_series.rolling(12)
            year_movav = np.array(year_window.mean().tolist()[11::])
            # Set new data as moving averages
            data_year = year_movav[0::12]
        elif time_data[i] == "season":
            data_month = [[] for i in range(len(time_ind[i]))]
            # Not crossing into the next year
            if not (0 in time_ind[i] and 12 in time_ind[i]):
                for j in range(len(time_ind[i])):
                    data_month[j] = np.array(data_sel[i][var_data[i]])\
                                    [time_ind[i][j]::12]
            else: # Crossing into the next year
                time_ind[i].sort()
                for j in range(len(time_ind[i])):
                    if j == 0 or time_ind[i][j] - time_ind[i][j-1] == 1:
                        data_month[j] = np.array(data_sel[i][var_data[i]])\
                                        [time_ind[i][j]+12::12]
                    else:
                        data_month[j] = np.array(data_sel[i][var_data[i]])\
                                        [time_ind[i][j]:-11:12]
            # Average over the months belonging to the season of interest            
            data_year = np.mean(np.array(data_month), axis=0)
        elif time_data[i] == "month":
            data_year = np.array(data_sel[i][var_data[i]])[time_ind[i]::12]
        
        # Moving average
        data_series = pd.Series(data_year)
        data_window = data_series.rolling(aggregation_time)
        data_movav = np.array(data_window.mean().tolist()
                              [(aggregation_time-1)::])
        
        # print(data_movav)
        
        # Select only relevant timesteps
        data_var[i] = data_movav[0::aggregation_time]
    
    return data_var
    

#%% SETTINGS

# List of all available variables
# index_list = ["amoc26", "sstatlspg", "sssatlspg", "mldatlspg", "maxsfspg"]
index_list = ["avsfspg", "sssatlspg", "mldatlspg", "subthspg", "rhoatlspg"]
domain_list = ["stdmld", "std", "std", "std", "std"]
var_data = ["msftbarot", "sos", "mlotst", "thetao", "rho" ]
plot_label = ["Streamfunction", "SSS", "MLD", "Subsurface temp.", "Density"]

# index_list = ["subthspg"]
# domain_list = ["std"]
# var_data = ["thetao"]
# plot_label = ["SF"]

# index_list = ["sssatlspg"]
# domain_list = ["std"]
# var_data = ["sos"]
# plot_label = ["SSS"]

# Set experiment id (only piControl atm)
exp_id = "piControl"
# Directory where the variables are stored
dir_path = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
            'CMIP6_indices/'
# Directory to save the figures
save_pcmci_fig = '/Users/3753808/Library/CloudStorage/' \
                 'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                 'PCMCI_results/'

# Directory to save the correlation check
save_check_fig = '/Users/3753808/Library/CloudStorage/' \
                 'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                 'Figs_SPG_PCMCI/Correlation_checks/'
                 
# # Settings to get all data
# time_data           = ["month"]
# time_ind            = [[0,1,2,3,4,5,6,7,8,9,10,11]]
# aggregation_time    = 1

# List of available models for all variables
index_path, index_sel, mod_all_list, mod_list = \
    variable_selection(index_list, domain_list) 

#%% ONE VARIABLE PLOTS
# Number of years to plot
yr = 6
# plot_lim = [-5,10]

# For each model
for mod in mod_list:
    # Only use the model versions of interest
    if mod[-8::] not in ['r1i1p1f1','r1i1p1f2']:
        continue
    
    # Get all data
    data_var = data_load(mod, index_list, index_path, index_sel, mod_all_list)
    # Print model and data length
    print(mod)
    print(np.min([len(data_var[i].time)/12 for i in range(len(index_list))]))
    
    # Plot timeseries
    # fig, axs = plt.subplots(2,len(index_list), figsize=(6,8))
    # fig.suptitle(mod)
    # # for i in range(len(index_list)):
    # # Full timeseries
    # axs[0].plot(data_var[0].time, data_var[0][var_data[0]])
    # # axs[0].set_ylim(plot_lim)
    # axs[0].set_ylabel(plot_label[0])
    # # Zoomed in part
    # axs[1].plot(data_var[0].time[0:(yr*12)], data_var[0][var_data[0]][0:(yr*12)])
    # # axs[1].set_ylim(plot_lim)
    # axs[1].set_ylabel(plot_label[0])

    # axs[0].grid()
    # axs[1].grid()
    # plt.tight_layout()
    # plt.show()

#%% PLOT TIMESERIES (FULL AND FIRST N YEARS)

# Number of years to plot
yr = 6
plot_lim = [None,[-1.5,2], [-700,2000], [-1.5,1.5], [-1,2]]

# For each model
for mod in mod_list:
    # Only use the model versions of interest
    if mod[-8::] not in ['r1i1p1f1','r1i1p1f2']:
        continue
    
    # Get all data
    data_var = data_load(mod, index_list, index_path, index_sel, mod_all_list)
    # Print model and data length
    print(mod)
    print([len(data_var[i].time) for i in range(len(index_list))])
    
    # Plot timeseries
    fig, axs = plt.subplots(2,len(index_list), figsize=(16,8))
    fig.suptitle(mod)
    for i in range(len(index_list)):
        # Full timeseries
        axs[0,i].plot(data_var[i].time, data_var[i][var_data[i]])
        axs[0,i].set_ylim(plot_lim[i])
        axs[0,i].set_ylabel(plot_label[i])
        # Zoomed in part
        axs[1,i].plot(data_var[i].time[0:(yr*12)], data_var[i][var_data[i]][0:(yr*12)])
        axs[1,i].set_ylim(plot_lim[i])
        axs[1,i].set_ylabel(plot_label[i])

        axs[0,i].grid()
        axs[1,i].grid()
    plt.tight_layout()
    plt.show()
    
#%% SCATTER PLOTS BETWEEN VARIABLES FOR SELECTED PERIOD
"""
PLOTS TO CHECK (LAG-1) CORRELATION BETWEEN THE VARIABLES. DOES IT MAKE SENSE?
"""

# Remove model with holes
# mod_list_arr = mod_list.tolist()
# mod_list_arr.remove('CESM2-WACCM-FV2_r1i2p2f1')

time_data = ["season", "season", "season", "season", "season"]
time_ind = [[0,1,2], [0,1,2], [0,1,2], [0,1,2], [0,1,2]]
aggregation_time = 1

plot_lim = [None,[-1.5,2], [-700,2000], [-1.5,1.5], [-1,2]]

# For each model
for mod in mod_list:
    # Only use the model versions of interest
    if mod[-8::] not in ['r1i1p1f1','r1i1p1f2']:
        continue
    print(mod)
    # Get all data
    data_var = data_load(mod, index_list, index_path, index_sel, mod_all_list)
    data_per = data_period(data_var, index_sel, var_data, aggregation_time, 
                           time_data, time_ind)
    
    # # Plot winter timeseries
    # fig, axs = plt.subplots(2,len(index_list), figsize=(16,8))
    # fig.suptitle(mod)
    # for i in range(len(index_list)):
    #     # Full timeseries
    #     axs[0,i].plot(data_per[i])
    #     axs[0,i].set_ylim(plot_lim[i])
    #     axs[0,i].set_ylabel(plot_label[i])
    #     # Zoomed in part
    #     axs[1,i].plot(data_per[i][0:yr])
    #     axs[1,i].set_ylim(plot_lim[i])
    #     axs[1,i].set_ylabel(plot_label[i])
    
    #     axs[0,i].grid()
    #     axs[1,i].grid()
    # plt.tight_layout()
    # plt.show()
    
    # Scatter plots of correlation
    fig, axs = plt.subplots(len(index_list),len(index_list), figsize=(20,20))
    fig.suptitle(mod + ", " + repr(data_per.shape[1]) + " years", fontsize=20)
    for i in range(len(index_list)):
        for j in range(len(index_list)):
            # Scatter lag-1 correlation
            axs[i,j].scatter(data_per[i,0:-1], data_per[j,1::], marker='.')
            # Scatter autocorrelation
            if j != i:
                axs[i,j].scatter(data_per[i], data_per[j], marker='.')
            # if j == i:
            #     axs[i,j].plot(np.arange(-1*10**11,1*10**11,1*10**10),
            #                   np.arange(-1*10**11,1*10**11,1*10**10), c='k')
            
            # axs[i,j].set_xlim(plot_lim[i])
            # axs[i,j].set_ylim(plot_lim[j])
            axs[i,j].set_xlabel(plot_label[i]+",t", fontsize = 14)
            axs[i,j].set_ylabel(plot_label[j]+",t+1", fontsize = 14)
            axs[i,j].grid()
        
    plt.tight_layout()
    # plt.savefig(save_check_fig + "Correlation_lag1_JFM_"+mod+".pdf")
    plt.show()
    
    # # Scatter plots of correlation
    # fig, axs = plt.subplots(len(index_list)-1,len(index_list)-1, figsize=(12,12))
    # fig.suptitle(mod)
    # for i in range(len(index_list)-1):
    #     for j in range(i,len(index_list)-1):
    #         axs[i,j].scatter(data_per[i], data_per[j+1])
    #         # axs[i,j].set_xlim(plot_lim[i])
    #         # axs[i,j].set_ylim(plot_lim[j+1])
    #         axs[i,j].set_xlabel(plot_label[i])
    #         axs[i,j].set_ylabel(plot_label[j+1])
    #         axs[i,j].grid()
    # plt.tight_layout()
    # plt.show()
    
    # # Scatter plots of lag-1 autocorrelation
    # fig, axs = plt.subplots(1,len(index_list), figsize=(16,6))
    # fig.suptitle(mod)
    # for i in range(len(index_list)):
    #         axs[i].scatter(data_per[i,0:-1], data_per[i,1::])
    #         # axs[i,j].set_xlim(plot_lim[i])
    #         # axs[i,j].set_ylim(plot_lim[j+1])
    #         axs[i].set_xlabel(plot_label[i]+",t")
    #         axs[i].set_ylabel(plot_label[i]+",t+1")
    #         axs[i].grid()
    # plt.tight_layout()
    # plt.show()


#%%

for i in range(1,2): #len(index_list)):
    print(index_list[i])
    # List of available models
    index_path, index_sel, mod_all_list, mod_list = \
        variable_selection([index_list[i]])
    # print(mod_all_list)    
    
    # print(["amoc_m","amoc_y"][i])
    # # List of available models
    # index_path, index_sel, mod_all_list, mod_list = \
    #     variable_selection([["amoc_m","amoc_y"][i]])
    # # print(mod_all_list)  
    
    # Preprocessing
    len_list = np.zeros(len(mod_list))
    for m in range(len(mod_list)):
        # print(mod_list[m])
        data_var = data_check(mod_list[m], [index_list[i]], index_path, 
                              index_sel, mod_all_list)
        len_list[m] = len(data_var.time)
        # Print model and data length
        print(mod_list[m])
        print(int(len_list[m]))
        # try:
        #     if any(np.isnan(data_var.msftyz)):
        #         print("NaN")
        #         print(data_var.msftyz)
        
        # except Exception:
        #     print("msftmz")
        #     continue
        
        
    # for ll in len_list:
    #     print(int(ll))
    # for mod in mod_list:
    #     print(mod)

#%% TIMESERIES CHECK
"""
THINK ABOUT HOW TO DO THIS EFFICIENTLY AND E.G. GET MULTIPLE FIGURES IN ONE
PLOT. HOW TO DEAL WITH THE DIFFERENT VARIABLE NAMES?
"""
yr = 10

fig = plt.figure()
plt.title(mod_list[m])
plt.plot(data_var.time, data_var.tos)
plt.ylabel("SST")
plt.grid()
plt.show()


fig = plt.figure()
plt.title(mod_list[m])
plt.plot(data_var.time[0:(yr*12)], data_var.tos[0:(yr*12)])
plt.ylabel("SST")
plt.grid()
plt.show()














































