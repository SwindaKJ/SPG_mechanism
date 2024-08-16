#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:31:52 2024

@author: S.K.J. Falkena (s.k.j.falkena@uu.nl)

PCMCI+ applied to the relevant SPG variables. 
IMPORTANT: USE THE + ONE FOR DIRECT LINKS!

"""


#%% IMPORT

import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# To download tigramite: https://github.com/jakobrunge/tigramite
# Check out the tutorials there
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite.causal_effects import CausalEffects

import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import dcor

#%% FUNCTIONS

### Data Preparation ###

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
    mod_list_var : A list of all available models for each of the variables.
    mod_list : A list of the models that are available for all variables.

    """
    # Initialise
    index_path = [[] for i in range(len(index_selection))]
    index_list = [[] for i in range(len(index_selection))]
    mod_list_var = [[] for i in range(len(index_selection))]
    # For each variable
    for i in range(len(index_selection)):
        # Set the path to the directory where the data is stored
        index_path[i] = os.path.join(dir_path, index_selection[i], dom[i], exp_id)
        # Get a list of files in that directory
        index_list[i] = [x for x in os.listdir(index_path[i]) if x[0] == 'C']
        index_list[i].sort()
        # Match the models (should be the same, but he)
        mod_list_var[i] = np.array([filename.split('.')[1] + "_" 
                                    + filename.split('.')[3] 
                                    for filename in index_list[i]])
    
    # Select models which are available for all variables
    mod_list = np.copy(mod_list_var[0])
    for i in range(1, len(index_selection)):
        mod_temp = np.array([mod for mod in mod_list_var[i] if mod in mod_list])
        mod_list = np.copy(mod_temp)
    
    return index_path, index_list, mod_list_var, mod_list

def remove_doublemodels(mod_list):
    """
    Remove double models based on a reasonable ordering after 

    Parameters
    ----------
    mod_list : A list of models with version at the end.

    Returns
    -------
    A list of models with double versions removed.

    """
    for i in range(0,len(mod_list)):
        if mod_list[i][-8::] not in ['r1i1p1f1','r1i1p1f2']:
            mod_list[i] = 0
        if i > 0 and mod_list[i][0:-9] == mod_list[i-1][0:-9]:
            mod_list[i] = 0
    return [i for i in mod_list if i != '0']

def data_prep(mod, index_selection, index_path, index_sel, mod_list_var, 
              aggregation_time, time_data, time_ind=None, detrend=False):
    """
    Load the data for each model (all variables)

    Parameters
    ----------
    mod : The model.
    index_path : A list of the directory paths to the files in which each of 
        the variables is stored.
    index_list : A list of all the files in the variable directory (models).
    mod_list_var : A list of all available models for each of the variables.
    aggregation_time : The time over which to aggregate the data (years).
    time_data : A list of the types of data, annual, seasonal or monthly mean.
    time_ind : Optional. Indicate the month(s) to consider for seasonal and 
        monthly data The default is None.

    Returns
    -------
    data_var : A list containing the timeseries of each of the variables.

    """
    
    # Create lists to store data arrays and info
    data_xr_list = [[] for i in range(len(index_sel))]
    var_data = [[] for i in range(len(index_sel))]
    len_data = np.zeros(len(index_sel))
    trend_slope = np.zeros(len(index_sel))
    # Load data + information
    for i in range(len(index_sel)):
        # Select file names
        file = np.array(index_sel[i])[mod_list_var[i]==mod][0]
        # Create path to file
        filepath = os.path.join(index_path[i], file)
        # Load dataset
        dataset = Dataset(filepath, mode='r')
        # Transfer to xarray
        data_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(dataset))
        # Store in list
        data_xr_list[i] = data_xr
        # Get names of variables
        var_data[i] = list(data_xr.keys())[0]
        # Get list of lengths
        len_data[i] = len(data_xr.time)
    
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
        
        # Compute (normalised) anomaly data
        data_var_anom = (data_year - np.mean(data_year)) / np.std(data_year)
        
        # If detrending
        if detrend:
            print("Detrending")
            # De-trend (linear)
            time_array = np.arange(0,len(data_var_anom)).reshape((-1, 1))
            model_lr = LinearRegression().fit(time_array, data_var_anom)
            # Subtract linear fit to data
            data_var[i] = data_var_anom - model_lr.predict(time_array)
            # Get slope of fit
            trend_slope[i] = model_lr.coef_    
        else:
            data_var[i] = data_var_anom
        
        # fig = plt.figure()
        # plt.title(mod + ", " + index_selection[i])
        # plt.plot(time_array, data_var_anom)
        # plt.plot(time_array, data_var)
        # plt.grid()
        # plt.show()
    
    return data_var, trend_slope


### Causal discovery ###

def data_discovery(data_var, index_selection, max_lag=10, alpha=0.05, msr='pc',
                   pcmci_method = 'pcmci+'):
    """
    Run PCMCI for the selected variables and choices of data.

    Parameters
    ----------
    data_var : A list containing the timeseries of each of the variables.
    index_selection : The names of the variables.
    max_lag : The maximum lag to consider in the network (in years). 
        The default is 10.
    alpha : The significance level for inclusion of links in the network. 
        The default is 0.1.
    msr : The conditional independence test to use. Options are 'pc' (partial 
        correlation), 'gpdc' (Gaussian process distance correlation) and 'cmi'
        (conditional mutual information).
        The default is 'pc'.
    pcmci_method : Indicates which version of PCMCI to use. Options are 'pcmci+'
        and 'pcmci'.
        The default is 'pcmci+'.

    Returns
    -------
    dataframe : The dataframe created for the PCMCI algorihtm.
    pcmci: The PCMCI algorihtm
    results : The PCMCI results.
    names : The names of the variables.
    corr : The correlation and corresponding p-values between the variables.
    parcorr : The partial correlation (or other output metric) and 
        corresponding p-value between the variables.

    """
    # Get data as array
    data = np.array(data_var).T
    print(data.shape)
    
    ##### PCMCI #####
    # Get dimensions
    T, N = data.shape
    # Initialize dataframe object, specify time axis and variable names
    names = index_selection
    dataframe = pp.DataFrame(data, 
                             datatime = {0:np.arange(len(data))}, 
                             var_names=names)
    # Set measure to use
    if msr == 'pc': # Partial correlation
        ci_test = ParCorr(significance='analytic')
    elif msr == 'rpc': # Gaussian process distance correlation
        ci_test = RobustParCorr(significance='analytic')
    elif msr == 'gpdc': # Gaussian process distance correlation
        ci_test = GPDC(significance='analytic', gp_params=None)
    elif msr == 'cmi': # Conditional mutual information
        ci_test = CMIknn(significance='shuffle_test')
    # Initialize PCMCI
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=ci_test,
        verbosity=1)
    # Run PCMCI
    pcmci.verbosity = 1
    # Get output
    if pcmci_method == 'pcmci+':
        results = pcmci.run_pcmciplus(tau_max=max_lag,pc_alpha=alpha)
    elif pcmci_method == 'pcmci':
        results = pcmci.run_pcmci(tau_max=max_lag,pc_alpha=None, 
                                  alpha_level=alpha)
    
    # Get correlations
    correlations = pcmci.get_lagged_dependencies(tau_max=max_lag)
    
    # Dimensions: parcorr & p-value, var1, var2, lag
    corr = np.array([correlations['val_matrix'], correlations['p_matrix']])
    parcorr = np.array([results['val_matrix'], results['p_matrix']])
    
    return dataframe, pcmci, results, names, corr, parcorr


### Causal effect ###


### Plotting ###

def plot_graphs(results, names, mod, ind=1, save_fig=False, save_name=None, 
                save_pcmci_fig="/Users/3753808/Library/CloudStorage/" \
                    "OneDrive-UniversiteitUtrecht/Code/Tipping_links/" \
                    "PCMCI_results/"):
    """
    Plot the graph and time series graph of the PCMCI result and save the plots
    if desired.

    Parameters
    ----------
    results : The PCMCI results.
    names : The names of the variables.
    ind : Indicates which plots to give, use 0 for only graph.
        The default is 1, giving also the timeseries graph.
    save_fig : Optional. Indicate when to save the figures. 
        The default is False.
    save_name : The name under which to save the figures.
    save_pcmci_fig : The folder in which to save the figures.

    Returns
    -------
    None.

    """
    # Name the sub-folder after the included variables
    save_folder = names[0]
    for i in range(1,len(names)):
        save_folder = save_folder+"_"+names[i]
    # Plot graph
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        ); plt.title(mod)
    # Save
    if save_fig == True:
        plt.savefig(save_pcmci_fig + save_name+"_resultsgraph.pdf", 
                    format='pdf', dpi=1200)
    else:
        plt.show()
    
    # If want both plots (standard)
    if ind == 1:
        try:
            # Plot time series graph    
            tp.plot_time_series_graph(
                figsize=(6, 4),
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=names,
                link_colorbar_label='MCI',
                ); plt.title(mod)
            # Save
            if save_fig == True:
                plt.savefig(save_pcmci_fig + save_name+"_timegraph.pdf", 
                            format='pdf', dpi=1200)
            else:
                plt.show()
        except:
            pass
    
    return

#%% SETTINGS

# Set experiment id (only piCOntrol atm)
exp_id = "piControl"
# Directory where the variables are stored
dir_path = '/Users/3753808/Library/CloudStorage/' \
            'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
            'CMIP6_indices/'
# Directory to save the data
dir_save_data = '/Users/3753808/Library/CloudStorage/' \
                'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                'PCMCI_output/'
# Directory to save the figures
dir_save_fig = '/Users/3753808/Library/CloudStorage/' \
                'OneDrive-UniversiteitUtrecht/Code/Tipping_links/' \
                'Figs_SPG_PCMCI/'

# Set indices to consider (all)
index_selection     = ["avsfspg", "sssatlspg", "mldatlspg", "subthspg", "rhoatlspg"]
domain_selection    = ["stdmld", "std", "std", "std", "std"]
time_data           = ["season", "season", "season", "season", "season"]
time_ind            = [[0,1,2], [0,1,2], [0,1,2], [0,1,2], [0,1,2]]

# index_selection     = ["avsfspg", "sssatlspg", "mldatlspg", "subthspg"]
# domain_selection    = ["stdmld", "std", "std", "std"]
# time_data           = ["season", "season", "season", "season"]
# time_ind            = [[0,1,2], [0,1,2], [0,1,2], [0,1,2]]

aggregation_time    = 1
max_lag             = 10
minlen              = 98


#%% PREPROCESSING

# List of available models
index_path, index_sel, mod_list_var, mod_list = \
    variable_selection(index_selection, domain_selection) 
# Remove other versions
mod_list = remove_doublemodels(mod_list)

# Preprocessing
data_var = [[] for m in range(len(mod_list))]
trend_slope = np.zeros((len(mod_list), len(index_selection)))
for m in range(len(mod_list)):
    # Get data
    data_var[m], trend_slope[m] = data_prep(mod_list[m], index_selection, 
                                            index_path, index_sel, mod_list_var, 
                                            aggregation_time, time_data, 
                                            time_ind, False)

#%%% CAUSAL DISCOVERY CHOICES

metric      = 'rpc'
pcmci_meth  = 'pcmci'
alpha_lev   = 0.05


#% CAUSAL DISCOVERY: ALL VARIABLES
"""
Discover the network using all variables of interest.

"""

# Initialise correlations, partial correlations and graphs
corr_all = np.empty((len(mod_list),2,len(index_selection),len(index_selection), 
                 max_lag+1))
parcorr_all = np.empty((len(mod_list),2,len(index_selection),len(index_selection), 
                 max_lag+1))
graph_list_all = [[] for i in range(len(mod_list))]

# For each model
for m in range(len(mod_list)):
    # Run PCMCI
    dataframe, pcmci, results, names, corr_all[m], parcorr_all[m] = \
        data_discovery(data_var[m], index_selection, max_lag, alpha=alpha_lev, 
                        msr=metric, pcmci_method=pcmci_meth)
    # Store graph
    graph_list_all[m] = results['graph']
    # Plot graph
    # plot_graphs(results, names, mod_list[m], ind=0)

# To array
graph_all = np.array(graph_list_all)

# Save results
# np.savez(dir_save_data+"causaldiscovery_norho_allvariables_"+pcmci_meth+"_"+metric+
#           "_alpha"+repr(alpha_lev)+"_lag"+repr(max_lag)+".npz", 
#           mod_list=mod_list, corr=corr_all, parcorr=parcorr_all, graph=graph_all)


#% CAUSAL DISCOVERY: ALL PAIRS
"""
Discover the network considering all pairs separately.

"""

# Initialise correlations, partial correlations and graphs
corr_2p = np.empty((len(mod_list),len(index_selection),len(index_selection),2,2,2,max_lag+1))
parcorr_2p = np.empty((len(mod_list),len(index_selection),len(index_selection),2,2,2,max_lag+1))
graph_list_theoryp = [[[[] for j in range(len(index_selection))]
                 for i in range(len(index_selection))] 
                for m in range(len(mod_list))]

# Store the dataframes
dataframe_2p = [[[[] for j in range(len(index_selection))]
                 for i in range(len(index_selection))] 
                for m in range(len(mod_list))]

# For each model
for m in range(len(mod_list)):
    # For each set variables
    for i in range(len(index_selection)):
        for j in range(len(index_selection)):
            if j > i:
                dataframe_2p[m][i][j], pcmci, results, names, corr_2p[m,i,j], parcorr_2p[m,i,j] = \
                    data_discovery(np.array([data_var[m][i],data_var[m][j]]), 
                                   [index_selection[i],index_selection[j]], 
                                   max_lag, alpha=alpha_lev, msr=metric, 
                                   pcmci_method=pcmci_meth)
                # Store graph
                graph_list_theoryp[m][i][j] = results['graph']
                # Plot graph
                # plot_graphs(results, names, mod_list[m], ind=0)

# Mirror and add diagonal to allow for conversion to array
# For each model
for m in range(len(mod_list)):
    # For each set variables
    for i in range(len(index_selection)):
        for j in range(len(index_selection)):
            if j > i:
                graph_list_theoryp[m][j][i] =  graph_list_theoryp[m][i][j]
            elif j == i:
                graph_list_theoryp[m][j][i] = \
                    np.array([[['' for i in range(max_lag+1)],
                               ['' for i in range(max_lag+1)]],
                              [['' for i in range(max_lag+1)],
                               ['' for i in range(max_lag+1)]]])
# To array
graph_2p = np.array(graph_list_theoryp)

# Save results
# np.savez(dir_save_data+"causaldiscovery_norho_allpairs_"+pcmci_meth+"_"+metric+
#           "_alpha"+repr(alpha_lev)+"_lag"+repr(max_lag)+".npz", 
#           mod_list=mod_list, corr=corr_2p, parcorr=parcorr_2p, graph=graph_2p)


#% CAUSAL DISCOVERY: SETS OF VARIABLES FOLLOWING THEORY
"""
Discover the network considering the variables following the theoretical
framwework.

"""

# Initialise correlations, partial correlations and graphs
corr_theory = np.empty((len(mod_list),len(index_selection),2,3,3,max_lag+1))
parcorr_theory = np.empty((len(mod_list),len(index_selection),2,3,3,max_lag+1))
graph_list_theory = [[[] for i in range(len(index_selection))] 
              for m in range(len(mod_list))]

# Store the dataframes
dataframe_theory = [[[] for i in range(len(index_selection))] 
              for m in range(len(mod_list))]

# For each model
for m in range(len(mod_list)):
    # For each set of subsequent variable in theory
    for i in range(len(index_selection)):
        # Run PCMCI
        if i == 0:
            dataframe_theory[m][i], pcmci, results, names, corr_theory[m,i], parcorr_theory[m,i] = \
                data_discovery(np.array([data_var[m][-1],data_var[m][0],data_var[m][1]]), 
                               [index_selection[-1],index_selection[0],index_selection[1]], 
                               max_lag, alpha=alpha_lev, msr=metric, 
                               pcmci_method=pcmci_meth)
        elif i+1 < len(index_selection):
            dataframe_theory[m][i], pcmci, results, names, corr_theory[m,i], parcorr_theory[m,i] = \
                data_discovery(data_var[m][i-1:i+2], index_selection[i-1:i+2], 
                               max_lag, alpha=alpha_lev, msr=metric, 
                               pcmci_method=pcmci_meth)
        elif i+1 == len(index_selection):
            dataframe_theory[m][i], pcmci, results, names, corr_theory[m,i], parcorr_theory[m,i] = \
                data_discovery(np.array([data_var[m][-2],data_var[m][-1],data_var[m][0]]), 
                               [index_selection[-2],index_selection[-1],index_selection[0]], 
                               max_lag, alpha=alpha_lev, msr=metric, 
                               pcmci_method=pcmci_meth)
        # Store graph
        graph_list_theory[m][i] = results['graph']
        # Plot graph
        # plot_graphs(results, names, mod_list[m], ind=0)
    
    # Save for one model
    # np.savez(dir_save_data+"/gpdc_models/causaldiscovery_gpdc_theory_"
    #          +mod_list[m]+".npz", mod=mod_list[m], corr=corr_theory[m], 
    #          parcorr=parcorr_theory[m], graph=np.array(graph_list_theory[m]))

# To array
graph_theory = np.array(graph_list_theory)

# Save results
# np.savez(dir_save_data+"causaldiscovery_norho_theory_"+pcmci_meth+"_"+metric+
#           "_alpha"+repr(alpha_lev)+"_lag"+repr(max_lag)+".npz", 
#           mod_list=mod_list, corr=corr_theory, parcorr=parcorr_theory, 
#           graph=graph_theory)



#%% CAUSAL EFFECT: THEORY

estimated_causal_effects = np.zeros((len(mod_list), len(index_selection)))

# Theoretical link between two variables
theory_graph = np.array([['','-->',''],
                         ['<--','','-->'],
                         ['','<--','']])

# For each model
for m in range(len(mod_list)):
    for i in range(len(index_selection)):
        # Set causal effect stuff
        causal_effect = CausalEffects(theory_graph, graph_type='dag', 
                                      X=[(1,0)], Y=[(2,0)], S=None, 
                                      hidden_variables=None, verbosity=1)
        
        # Fit causal effect model from observational data
        causal_effect.fit_total_effect(
                dataframe=dataframe_theory[m][i], 
                estimator=LinearRegression(),
                adjustment_set='optimal',
                conditional_estimator=None,  
                data_transform=None,
                mask_type=None,
                )
        
        # Predict effect of interventions
        intervention_data = 1.*np.ones((1, 1))
        
        estimated_causal_effects[m,i] = causal_effect.predict_total_effect( 
                intervention_data=intervention_data,
        #         conditions_data=conditions_data,
                )

#%% CAUSAL EFFECT: SETS OF VARIABLES INCL. LAG (BOTH)

estimated_causal_effects_lag = np.zeros((len(mod_list),len(index_selection),2))

# Theoretical link between two variables
theory_graph_lag = np.array([[['','-->'],['-->','-->'],['','']],
                              [['<--',''],['','-->'],['-->','-->']],
                              [['',''],['<--',''],['','-->']]])

# theory_graph_lag = np.array([[['','-->'],['','-->'],['','']],
#                               [['',''],['','-->'],['-->','-->']],
#                               [['',''],['<--',''],['','-->']]])


# For each model
for m in range(len(mod_list)):
    for i in range(len(index_selection)):
        # Set causal effect stuff
        causal_effect_lag = CausalEffects(theory_graph_lag, 
                                          graph_type='stationary_dag', 
                                          X=[(1,0),(1,-1)], Y=[(2,0)], S=None, 
                                          hidden_variables=None, verbosity=1)
        
        # Fit causal effect model from observational data
        causal_effect_lag.fit_total_effect(
                dataframe=dataframe_theory[m][i], 
                estimator=LinearRegression(),
                adjustment_set='optimal',
                conditional_estimator=None,  
                data_transform=None,
                mask_type=None,
                )
        
        # Predict effect of interventions
        intervention_data = [np.array([[1,0]]), np.array([[0,1]])]
        
        for j in range(len(intervention_data)):
            estimated_causal_effects_lag[m,i,j] = causal_effect_lag.predict_total_effect( 
                    intervention_data=intervention_data[j],
            #         conditions_data=conditions_data,
                    )

#%% CAUSAL EFFECT: SETS OF VARIABLES from #models

# estimated_causal_effects_list = np.zeros((len(mod_list),len(index_selection),2))
estimated_causal_effects_list = np.zeros((len(mod_list),len(index_selection),7))

# Theoretical link between three variables following theory: 0.01
# theory_graph_list = [np.array([[['','-->'],['','-->'],['','']],
#                               [['',''],['','-->'],['','-->']],
#                               [['',''],['',''],['','-->']]]),
#                       np.array([[['','-->'],['','-->'],['','']],
#                                 [['',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->','-->'],['','']],
#                                 [['<--',''],['','-->'],['-->','']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->',''],['','']],
#                                 [['<--',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->','-->'],['','']],
#                                 [['<--',''],['','-->'],['','-->']],
#                                 [['',''],['',''],['','-->']]])]
# X_list = [[(1,-1)],
#           [(1,0),(1,-1)],
#           [(1,0)],
#           [(1,0),(1,-1)],
#           [(1,-1)]]
# intervention_list = [[1.*np.ones((1, 1))],
#                       [np.array([[1,0]]), np.array([[0,1]])],
#                       [1.*np.ones((1, 1))],
#                       [np.array([[1,0]]), np.array([[0,1]])],
#                       [1.*np.ones((1, 1))]]

# Theoretical link between three variables following theory: 0.05
# theory_graph_list = [np.array([[['','-->'],['-->','-->'],['','']],
#                               [['<--',''],['','-->'],['','-->']],
#                               [['',''],['',''],['','-->']]]),
#                       np.array([[['','-->'],['','-->'],['','']],
#                                 [['',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->','-->'],['','']],
#                                 [['<--',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->','-->'],['','']],
#                                 [['<--',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]]),
#                       np.array([[['','-->'],['-->','-->'],['','']],
#                                 [['<--',''],['','-->'],['-->','-->']],
#                                 [['',''],['<--',''],['','-->']]])]
# X_list = [[(1,-1)],
#           [(1,0),(1,-1)],
#           [(1,0),(1,-1)],
#           [(1,0),(1,-1)],
#           [(1,0),(1,-1)]]
# intervention_list = [[1.*np.ones((1, 1))],
#                       [np.array([[1,0]]), np.array([[0,1]])],
#                       [np.array([[1,0]]), np.array([[0,1]])],
#                       [np.array([[1,0]]), np.array([[0,1]])],
#                       [np.array([[1,0]]), np.array([[0,1]])]]


# Theoretical link between three variables following theory: 0.05, lag 5
theory_graph_list = [np.array([[['','-->','','','',''],['-->','-->','','','',''],['','','','','','']],
                              [['<--','','','','',''],['','-->','-->','-->','-->','-->'],['','-->','','','','']],
                              [['','','','','',''],['','','','','',''],['','-->','','','','']]]),
                      np.array([[['','-->','-->','-->','-->','-->'],['','-->','','','',''],['','','','','','']],
                                [['','','','','',''],['','-->','','','',''],['-->','-->','','','','']],
                                [['','','','','',''],['<--','','','','',''],['','-->','','','','']]]),
                      np.array([[['','-->','','','',''],['-->','-->','','','',''],['','','','','','']],
                                [['<--','','','','',''],['','-->','','','',''],['-->','-->','','','','']],
                                [['','','','','',''],['<--','','','','',''],['','-->','-->','','','']]]),
                      np.array([[['','-->','','','',''],['-->','-->','','','',''],['','','','','','']],
                                [['<--','','','','',''],['','-->','-->','','',''],['-->','-->','','','','']],
                                [['','','','','',''],['<--','','','','',''],['','-->','','','','']]]),
                      np.array([[['','-->','-->','','',''],['-->','-->','','','',''],['','','','','','']],
                                [['<--','','','','',''],['','-->','','','',''],['-->','-->','','','','']],
                                [['','','','','',''],['<--','','','','',''],['','-->','-->','-->','-->','-->']]])]
X_list = [[(1,-1), (2,-1)],
          [(1,0),(1,-1), (2,-1)],
          [(1,0),(1,-1), (2,-1),(2,-2)],
          [(1,0),(1,-1), (2,-1)],
          [(1,0),(1,-1), (2,-1),(2,-2),(2,-3),(2,-4),(2,-5)]]
intervention_list = [[np.array([[1,0]]), np.array([[0,1]])],
                      [np.array([[1,0,0]]), np.array([[0,1,0]]), np.array([[0,0,1]])],
                      [np.array([[1,0,0,0]]), np.array([[0,1,0,0]]), 
                        np.array([[0,0,1,0]]), np.array([[0,0,0,1]])],
                      [np.array([[1,0,0]]), np.array([[0,1,0]]), np.array([[0,0,1]])],
                      [np.array([[1,0,0,0,0,0,0]]), np.array([[0,1,0,0,0,0,0]]), 
                        np.array([[0,0,1,0,0,0,0]]), np.array([[0,0,0,1,0,0,0]]),
                        np.array([[0,0,0,0,1,0,0]]), np.array([[0,0,0,0,0,1,0]]), 
                        np.array([[0,0,0,0,0,0,1]])]]


# No Rho
# theory_graph_list = [np.array([[['','-->','-->'],['','-->','-->'],['','','']],
#                                 [['','',''],['','-->','-->'],['','-->','']],
#                                 [['','',''],['','',''],['','-->','']]]),
#                         np.array([[['','-->','-->'],['','-->',''],['','','']],
#                                   [['','',''],['','-->',''],['-->','-->','']],
#                                   [['','',''],['<--','',''],['','-->','']]]),
#                         np.array([[['','-->',''],['-->','-->',''],['','','']],
#                                   [['<--','',''],['','-->',''],['-->','-->','']],
#                                   [['','',''],['<--','',''],['','-->','-->']]]),
#                         np.array([[['','-->',''],['-->','-->',''],['','','']],
#                                   [['<--','',''],['','-->','-->'],['','-->','-->']],
#                                   [['','',''],['','',''],['','-->','-->']]])]
# X_list = [[(1,-1), (2,-1)],
#           [(1,0),(1,-1), (2,-1)],
#           [(1,0),(1,-1), (2,-1),(2,-2)],
#           [(1,-1),(1,-2), (2,-1),(2,-2)]]
# intervention_list = [[np.array([[1,0]]), np.array([[0,1]])],
#                      [np.array([[1,0,0]]), np.array([[0,1,0]]), np.array([[0,0,1]])],
#                      [np.array([[1,0,0,0]]), np.array([[0,1,0,0]]), 
#                       np.array([[0,0,1,0]]), np.array([[0,0,0,1]])],
#                      [np.array([[1,0,0,0]]), np.array([[0,1,0,0]]), 
#                       np.array([[0,0,1,0]]), np.array([[0,0,0,1]])]]

# No Rho, up to lag-5
# theory_graph_list = [np.array([[['','-->','-->','','',''],['','-->','-->','','',''],['','','','','','']],
#                                 [['','','','','',''],['','-->','-->','-->','-->','-->'],['','-->','','','','']],
#                                 [['','','','','',''],['','','','','',''],['','-->','-->','','','']]]),
#                         np.array([[['','-->','-->','-->','-->','-->'],['','-->','','','',''],['','','','','','']],
#                                   [['','','','','',''],['','-->','-->','','',''],['-->','-->','','','','']],
#                                   [['','','','','',''],['<--','','','','',''],['','-->','','','','']]]),
#                         np.array([[['','-->','-->','','',''],['-->','-->','','','',''],['','','','','','']],
#                                   [['<--','','','','',''],['','-->','','','',''],['-->','-->','','','','']],
#                                   [['','','','','',''],['<--','','','','',''],['','-->','-->','','','']]]),
#                         np.array([[['','-->','','','',''],['-->','-->','','','',''],['','','','','','']],
#                                   [['<--','','','','',''],['','-->','-->','','',''],['','-->','-->','','','']],
#                                   [['','','','','',''],['','','','','',''],['','-->','-->','-->','-->','-->']]])]
# X_list = [[(1,-1), (2,-1),(2,-2)],
#           [(1,0),(1,-1), (2,-1)],
#           [(1,0),(1,-1), (2,-1),(2,-2)],
#           [(1,-1),(1,-2), (2,-1),(2,-2),(2,-3),(2,-4),(2,-5)]]
# intervention_list = [[np.array([[1,0,0]]), np.array([[0,1,0]]), np.array([[0,0,1]])],
#                      [np.array([[1,0,0]]), np.array([[0,1,0]]), np.array([[0,0,1]])],
#                      [np.array([[1,0,0,0]]), np.array([[0,1,0,0]]), 
#                       np.array([[0,0,1,0]]), np.array([[0,0,0,1]])],
#                      [np.array([[1,0,0,0,0,0,0]]), np.array([[0,1,0,0,0,0,0]]), 
#                       np.array([[0,0,1,0,0,0,0]]), np.array([[0,0,0,1,0,0,0]]),
#                       np.array([[0,0,0,0,1,0,0]]), np.array([[0,0,0,0,0,1,0]]), 
#                       np.array([[0,0,0,0,0,0,1]])]]

# For each model
for m in range(len(mod_list)):
    for i in range(len(index_selection)):
        # Set causal effect stuff
        causal_effect_lag = CausalEffects(theory_graph_list[i], 
                                          graph_type='stationary_dag', 
                                          X=X_list[i], Y=[(2,0)], S=None, 
                                          hidden_variables=None, verbosity=1)
        
        # Fit causal effect model from observational data
        causal_effect_lag.fit_total_effect(
                dataframe=dataframe_theory[m][i], 
                estimator=LinearRegression(),
                adjustment_set='optimal',
                conditional_estimator=None,  
                data_transform=None,
                mask_type=None,
                )
        
        # Predict effect of interventions
        intervention_data = intervention_list[i]
        
        for j in range(len(intervention_data)):
            estimated_causal_effects_list[m,i,j] = causal_effect_lag.predict_total_effect( 
                    intervention_data=intervention_data[j],
            #         conditions_data=conditions_data,
                    )

#%% Scatter plot

link_names = ["SPG to SSS", "SSS to MLD", "MLD to SubT", "SubT to Rho",
              "Rho to SPG"]
col_modlist = ['tab:blue','tab:orange','tab:orange','tab:green','tab:green',
                'tab:green','tab:green','tab:red','tab:red','tab:purple',
                'tab:purple','tab:brown','tab:brown','tab:brown','tab:brown',
                'tab:brown','tab:pink','tab:pink','tab:gray','tab:gray',
                'tab:olive','tab:olive','gold','chocolate','tab:cyan','tab:cyan',
                'tab:cyan','black','lightgreen','lightgreen','navy','navy']
mark_modlist = ['o','o','X','o','X','v','P','o','X','o','X','o','X','v','P',
                'd','o','X','o','X','o','X','o','o','o','X','v','o','o','X',
                'o','X']

# link_names = ["SPG to SSS", "SSS to MLD", "MLD to SubT", 
#               "SubT to SPG"]
# col_modlist = ['tab:blue','tab:orange','tab:orange','tab:green','tab:green',
#                 'tab:green','tab:green','tab:red','tab:red','tab:purple',
#                 'tab:purple','tab:brown','tab:brown','tab:brown','tab:brown',
#                 'tab:brown','tab:brown','tab:pink','tab:pink','tab:gray','tab:gray',
#                 'tab:olive','tab:olive','gold','chocolate','tab:cyan','tab:cyan',
#                 'tab:cyan','black','lightgreen','lightgreen','navy','navy']
# mark_modlist = ['o','o','X','o','X','v','P','o','X','o','X','o','X','v','P',
#                 'd','*','o','X','o','X','o','X','o','o','o','X','v','o','o','X',
#                 'o','X']

col_modsig = np.array(col_modlist.copy())


# Get color when significant
sig_list = np.array([parcorr_theory[:,i,1,1,2,0] 
                      for i in range(len(index_selection))])
# allsig = np.array([all(sig_list[:,m] < p_thres) for m in range(len(mod_list))])
# col_modsig[allsig==False] = None

# Based on data being significant when not correcting for confounding factors
fig = plt.figure(figsize=(17,11))

# Scatter plots
for m in range(len(mod_list)):
    for i in range(len(index_selection)):
        # Split data in significant and non-significant
        col_modsig = np.array(col_modlist.copy())
        # Link focussed incl parent
        # col_modsig[parcorr_theory[:,i,1,1,2,0] > p_thres] = None
        # col_modsig[parcorr_theory_gpdc[:,i,1,1,2,0] > p_thres] = None
        # Includsing all variables
        # col_modsig[parcorr[:,1,i,(i+1)%len(index_selection),0] > p_thres] = None
        
        if i == 0:
            plt.scatter(np.arange(i-0.3,i+0.299,0.6/len(mod_list))[m], 
                        estimated_causal_effects[m,i], marker=mark_modlist[m],
                        edgecolors=col_modlist[m], facecolors=col_modsig[m],
                        label=mod_list[m], s=120)
        else:
            plt.scatter(np.arange(i-0.3,i+0.299,0.6/len(mod_list))[m], 
                        estimated_causal_effects[m,i], marker=mark_modlist[m],
                        edgecolors=col_modlist[m], facecolors=col_modsig[m],
                        s=120)

plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
plt.xticks(np.arange(len(index_selection)), link_names, fontsize=16)
plt.yticks(fontsize=14)
plt.xlim([-0.5, len(index_selection)-0.5])
plt.ylabel("Causal Effect", fontsize=16)
plt.grid()
plt.legend(fontsize=14, ncols=4, loc='upper right', bbox_to_anchor=(0.995, 1.34))
plt.tight_layout()
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha"+repr(alpha_lev)+"_alltheory.pdf")
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha"+repr(alpha_lev)+"_inclparent.pdf")
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha.pdf")
plt.show()

#%% Violin plot of causal effect

link_names = ["SPG to SSS", "SSS to MLD", "MLD to SubT", 
              "SubT to Rho", "Rho to SPG"]
col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Based on data being significant when not correcting for confounding factors
fig = plt.figure(figsize=(12,8))

for i in range(len(index_selection)):
    # Violin plot
    vio = plt.violinplot(estimated_causal_effects[:,i], positions = [i])
    # Scatter plots
    scat = plt.scatter((i)*np.ones(len(mod_list)), estimated_causal_effects[:,i], c='k')
    
    # Color of violin plots
    for pc in vio['bodies']:
        pc.set_color(col_list[i])
    for partname in ('cbars','cmins','cmaxes'):
        vio[partname].set_edgecolor(col_list[i])

plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
plt.xticks(np.arange(len(index_selection)), link_names, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([-0.5, len(index_selection)-0.5])
plt.grid()
# plt.savefig(dir_save_fig + "Causaleffect_baseline.pdf")
plt.show()

#%% LOAD DATA
"""
Load the data as saved for different approaches.

"""

# Set the significance level to consider
alpha_lev = 0.05

# List of options and methods
data_sel = ["allvariables", "allpairs", "theory"]
meth_list = ['pcmci', 'pcmci+']
metric_list = ['pc','rpc']

# Initialise lists with all the data
corr_all_list = [[[[] for k in range(2)] for j in range(2)] for i in range(3)]
parcorr_all_list = [[[[] for k in range(2)] for j in range(2)] for i in range(3)]
graph_list_all_list = [[[[] for k in range(2)] for j in range(2)] for i in range(3)]


# For all different sets of variables considerd in pcmci, load the data
for i in range(3):
    print(data_sel[i])
    for j in range(2):
        pcmci_meth = meth_list[j]
        for k in range(2):
            metric = metric_list[k]
            res_causal = np.load(dir_save_data+"causaldiscovery_"+data_sel[i]+"_"
                                  +pcmci_meth+"_"+metric+"_alpha"+repr(alpha_lev)
                                  +"_lag"+repr(max_lag)+".npz")
            # res_causal = np.load(dir_save_data+"causaldiscovery_norho_"+data_sel[i]+"_"
            #                       +pcmci_meth+"_"+metric+"_alpha"+repr(alpha_lev)
            #                       +"_lag"+repr(max_lag)+".npz")
            corr_all_list[i][j][k] = res_causal['corr']
            parcorr_all_list[i][j][k] = res_causal['parcorr']
            graph_list_all_list[i][j][k] = res_causal['graph']

# Convert to list of arrays
corr_all = [[] for i in range(3)]
parcorr_all = [[] for i in range(3)]
graph_all = [[] for i in range(3)]
for i in range(3):
    corr_all[i] = np.array(corr_all_list[i])
    parcorr_all[i] = np.array(parcorr_all_list[i])
    graph_all[i] = np.array(graph_list_all_list[i])

#% LINKS: WHICH MODELS AND COUNT
"""
Check the causal graph output and count the number of models for which each 
link is present.

"""

# Initialise lists for which links
where_right = [[] for i in range(3)]
where_left = [[] for i in range(3)]
where_contemp = [[] for i in range(3)]
where_unclear = [[] for i in range(3)]

# Initialise
links_r = [[] for i in range(3)]
links_l = [[] for i in range(3)]
links_c = [[] for i in range(3)]
links_u = [[] for i in range(3)]

# Identify each model with a link (in a direction)
for i in range(3):
    where_right[i] = np.where(graph_all[i] == '-->',1,0)
    where_left[i] = np.where(graph_all[i] == '<--',1,0)
    where_contemp[i] = np.where(graph_all[i] == 'o-o',1,0)
    where_unclear[i] = np.where(graph_all[i] == 'x-x',1,0)
    
    # Count the links (over all models)
    links_r[i] = np.sum(where_right[i], axis=2)
    links_l[i] = np.sum(where_left[i], axis=2)
    links_c[i] = np.sum(where_contemp[i], axis=2)
    links_u[i] = np.sum(where_unclear[i], axis=2)

#%% PLOT NUMBER OF LINKS FOR EACH METHOD

clist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink']

for i0 in range(3):
    fig, ax = plt.subplots(len(index_selection),len(index_selection), 
                           figsize=(10,10))
    for j in range(len(index_selection)):
        for k in range(len(index_selection)):
            # Set lines
            ax[j,k].hlines(5,-1,max_lag+1, color='k', linestyle='-')
            ax[j,k].hlines(10,-1,max_lag+1, color='k', linestyle=':')
            
            c_count = 0
            for i1 in range(2):
                for i2 in range(2):
                    if i0 == 0:
                        # Links including all 5 variables
                        ax[j,k].plot(np.arange(0,max_lag+1), 
                                     links_r[i0][i1,i2,j,k],'o', 
                                     c=clist[c_count])
                        ax[j,k].plot(np.arange(0,max_lag+1), 
                                     links_l[i0][i1,i2,j,k],'v', 
                                     c=clist[c_count])
                        ax[j,k].plot(np.arange(0,max_lag+1), 
                                     links_c[i0][i1,i2,j,k],'X', 
                                     c=clist[c_count])
                        ax[j,k].plot(np.arange(0,max_lag+1), 
                                     links_u[i0][i1,i2,j,k],'P', 
                                     c=clist[c_count])
                    if i0 == 1:
                        # Links focussing on pairwise links
                        if j == k:
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_r[i0][i1,i2,j,k,0,0],'o', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_l[i0][i1,i2,j,k,0,0],'v', 
                                         c=clist[c_count])
                        elif j < k:
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_r[i0][i1,i2,j,k,0,1],'o', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_l[i0][i1,i2,j,k,0,1],'v', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_c[i0][i1,i2,j,k,0,1],'X', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_u[i0][i1,i2,j,k,0,1],'P', 
                                         c=clist[c_count])
                        elif j >  k:
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_r[i0][i1,i2,j,k,1,0],'o', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_l[i0][i1,i2,j,k,1,0],'v', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_c[i0][i1,i2,j,k,1,0],'X', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_u[i0][i1,i2,j,k,1,0],'P', 
                                         c=clist[c_count])
                    if i0 == 2:
                        # Links for theoretical links
                        if j == k:
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_r[i0][i1,i2,j,1,1],'o', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_l[i0][i1,i2,j,1,1],'v', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_c[i0][i1,i2,j,1,1],'X', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_u[i0][i1,i2,j,1,1],'P', 
                                         c=clist[c_count])
                        if k == (j+1)%len(index_selection):
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_r[i0][i1,i2,j,1,2],'o', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_l[i0][i1,i2,j,1,2],'v', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_c[i0][i1,i2,j,1,2],'X', 
                                         c=clist[c_count])
                            ax[j,k].plot(np.arange(0,max_lag+1), 
                                         links_u[i0][i1,i2,j,1,2],'P', 
                                         c=clist[c_count])
                    # print(meth_list[i1])
                    # print(metric_list[i2])
                    c_count += 1
        # Links focussing on pairwise links
        # if j == i:
        #     for k in range(len(index_selection)):
        #         ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,k,0,0],'X', c=clist[k+2], markerfacecolor='none')
        # elif i < j:
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,j,0,1],'X', c='tab:green')
        # elif i > j:
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,j,1,0],'X', c='tab:green')
        # Links focussing on theoretical links
        # if j == i:
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory[i,0,0],'v', c='tab:orange')
        # if j == (i+1)%len(index_selection):
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory[i,0,1],'v', c='tab:orange')
        # Links focussing on theoretical links using gpdc
        # if j == i:
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory_gpdc[i,0,0],'^', c='tab:purple')
        # if j == (i+1)%len(index_selection):
        #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory_gpdc[i,0,1],'^', c='tab:purple')
        # Plot settings
            ax[j,k].set_xlim([-1,max_lag+1])
            ax[j,k].set_ylim([0,len(mod_list)+1])
            ax[j,k].grid()
            if j == len(index_selection)-1:
                ax[j,k].set_xlabel("Lag (years)", fontsize=8)
            if j == 0:
                ax[j,k].set_title(index_selection[k], fontsize=10)
            if k == 0:
                ax[j,k].set_ylabel(index_selection[j], fontsize=10)
    
    plt.tight_layout()
    # plt.savefig(dir_save_fig + "NrModelsWithLink_alpha"+repr(alpha_lev)+
    #             "_comparemethods_"+data_sel[i0]+".pdf")
    plt.show()

#%% COMPARE USED DATA
clist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink']
var_list = ["SPG", "SSS", "MLD", "SubT", "Rho"]

# PCMCI+, RobustParCorr
i1 = 1
i2 = 1

fig, ax = plt.subplots(len(index_selection),len(index_selection), 
                       figsize=(10,10))
for i0 in [2,0]:
    for j in range(len(index_selection)):
        for k in range(len(index_selection)):
            # Set lines
            ax[j,k].hlines(5,-1,max_lag+1, color='k', linestyle='-')
            ax[j,k].hlines(10,-1,max_lag+1, color='k', linestyle=':')
            
            if i0 == 0:
                # Links including all 5 variables
                ax[j,k].plot(np.arange(0,max_lag+1), 
                             links_r[i0][i1,i2,j,k],'o', 
                             c=clist[1])
                ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                             links_l[i0][i1,i2,j,k,0],'v', 
                             c=clist[1])
                ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                             links_c[i0][i1,i2,j,k,0],'X', 
                             c=clist[1])
                ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                             links_u[i0][i1,i2,j,k,0],'P', 
                             c=clist[1])
                ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                             links_r[i0][i1,i2,j,k,0] + links_u[i0][i1,i2,j,k,0] + links_c[i0][i1,i2,j,k,0],'s', 
                             c=clist[1])
            # if i0 == 1:
            #     # Links focussing on pairwise links
            #     if j == k:
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_r[i0][i1,i2,j,k,0,0],'o', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_l[i0][i1,i2,j,k,0,0],'v', 
            #                      c=clist[i0], alpha=0.5)
            #         # ax[j,k].plot(np.arange(0,max_lag+1), 
            #         #               links_l[i0][i1,i2,j,k,0,0] + links_r[i0][i1,i2,j,k,0,0],'s', 
            #         #               c=clist[i0])
            #     elif j < k:
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_r[i0][i1,i2,j,k,0,1],'o', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_l[i0][i1,i2,j,k,0,1],'v', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_c[i0][i1,i2,j,k,0,1],'X', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_u[i0][i1,i2,j,k,0,1],'P', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                       links_r[i0][i1,i2,j,k,0,1] + links_u[i0][i1,i2,j,k,0,1] + links_c[i0][i1,i2,j,k,0,1],'s', 
            #                       c=clist[i0], alpha=0.5)
            #     elif j >  k:
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_r[i0][i1,i2,j,k,1,0],'o', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_l[i0][i1,i2,j,k,1,0],'v', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_c[i0][i1,i2,j,k,1,0],'X', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                      links_u[i0][i1,i2,j,k,1,0],'P', 
            #                      c=clist[i0], alpha=0.5)
            #         ax[j,k].plot(np.arange(0,max_lag+1), 
            #                       links_r[i0][i1,i2,j,k,1,0] + links_u[i0][i1,i2,j,k,1,0] + links_c[i0][i1,i2,j,k,1,0],'s', 
            #                       c=clist[i0], alpha=0.5)
            if i0 == 2:
                # Links for theoretical links
                if j == k:
                    ax[(j+1)%len(index_selection),(j+1)%len(index_selection)].\
                        plot(np.arange(0.,max_lag+0.5), links_r[i0][i1,i2,j,2,2],
                             'o', c=clist[0])
                    # ax[j,k].plot(np.arange(0,max_lag+1), 
                    #              links_r[i0][i1,i2,j,1,1],'o', 
                    #              c=clist[0])
                    # ax[j,k].plot(np.arange(0,max_lag+1), 
                    #              links_l[i0][i1,i2,j,1,1],'v', 
                    #              c=clist[0])
                    # ax[j,k].plot(np.arange(0,max_lag+1), 
                    #              links_c[i0][i1,i2,j,1,1],'X', 
                    #              c=clist[0])
                    # ax[j,k].plot(np.arange(0,max_lag+1), 
                    #              links_u[i0][i1,i2,j,1,1],'P', 
                    #              c=clist[0])
                    # ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                    #               links_r[i0][i1,i2,j,1,1,0] + links_c[i0][i1,i2,j,1,1,0] + links_u[i0][i1,i2,j,1,1,0],'s', 
                    #               c=clist[i0])
                if k == (j+1)%len(index_selection):
                    ax[j,k].plot(np.arange(0,max_lag+1), 
                                 links_r[i0][i1,i2,j,1,2],'o', 
                                 c=clist[0])
                    ax[j,k].plot(np.arange(0,max_lag+1), 
                                  links_l[i0][i1,i2,j,1,2],'v', 
                                  c=clist[0])
                    ax[j,k].plot(np.arange(0,max_lag+1), 
                                  links_c[i0][i1,i2,j,1,2],'X', 
                                  c=clist[0])
                    ax[j,k].plot(np.arange(0,max_lag+1), 
                                  links_u[i0][i1,i2,j,1,2],'P', 
                                  c=clist[0])
                    ax[j,k].plot(np.arange(0,max_lag+1)[0], 
                                  links_r[i0][i1,i2,j,1,2,0] + links_c[i0][i1,i2,j,1,2,0] + links_u[i0][i1,i2,j,1,2,0],'s', 
                                  c=clist[0])
        # Plot settings
            ax[j,k].set_xlim([-1,max_lag+1])
            ax[j,k].set_ylim([0,len(mod_list)+1])
            ax[j,k].grid(True)
            if j == len(index_selection)-1:
                ax[j,k].set_xlabel("Lag (years)", fontsize=8)
            if j == 0:
                ax[j,k].set_title(var_list[k], fontsize=10)
            if k == 0:
                ax[j,k].set_ylabel(var_list[j], fontsize=10)
    
plt.tight_layout()
# plt.savefig(dir_save_fig + "NrModelsWithLink_alpha"+repr(alpha_lev)+
#             "_datacompare_"+meth_list[i1]+"_"+metric_list[i2]+".pdf")
plt.show()

#%% METHOD COMPARISON BASED ON LINKS

methods_names = ["PCMCI, ParCorr", "PCMCI, RobustParCorr", \
                 "PCMCI+, ParCorr", "PCMCI+, RobustParCorr"]

# lag_list = [[1], [0,1], [0], [0,1], [1]]
# lag_list = [[1], [0,1], [0,1], [0,1], [0,1]]
lag_list = [[1], [1], [0,1], [1], [1]]

fig, ax = plt.subplots(3,4, figsize=(18,10))

for i1 in range(2):
    for i2 in range(2):
        for i in range(len(index_selection)):
            # Split data in significant and non-significant
            # data_sig = estimated_causal_effects[parcorr_theory[:,i,1,1,2,0] < p_thres,i]
            # data_nonsig = estimated_causal_effects[parcorr_theory[:,i,1,1,2,0] > p_thres,i]
            
            ax[0,2*i1+i2].set_title(methods_names[2*i1+i2], fontsize = 14)
            
            # if i1 == 0:
            #     where_check = [sum(x) for x in zip(where_right, where_contemp)]
            # elif i1 == 1:
            where_check = [sum(x) for x in zip(where_right, where_contemp, where_unclear)]
            
            # For each of the lags
            for lag_ind in range(len(lag_list[i])):
                # Get causal effect and lag
                est_ce = estimated_causal_effects_list[:,:,lag_ind]
                lg = lag_list[i][lag_ind]
                
                data_sig = [[] for j in range(3)]
                data_nonsig = [[] for j in range(3)]
                
                # All 5 variables
                data_sig[0] = est_ce[where_check[0][i1,i2,:,i,(i+1)%len(index_selection),lg] == 1,i]
                data_nonsig[0] = est_ce[where_check[0][i1,i2,:,i,(i+1)%len(index_selection),lg] == 0,i]
                
                # Pairs
                data_sig[1] = est_ce[where_check[1][i1,i2,:,i,(i+1)%len(index_selection),0,1,lg] == 1,i]
                data_nonsig[1] = est_ce[where_check[1][i1,i2,:,i,(i+1)%len(index_selection),0,1,lg] == 0,i]
                
                # Theory
                data_sig[2] = est_ce[where_check[2][i1,i2,:,i,1,2,lg] == 1,i]
                data_nonsig[2] = est_ce[where_check[2][i1,i2,:,i,1,2,lg] == 0,i]
            
                for j in range(3):
                    # Violin plots
                    if not len(data_sig[j]) == 0:
                        vio_sig = ax[j,2*i1+i2].violinplot(data_sig[j], positions = [i-0.3+lg*0.4])
                        # Color of violin plots
                        for pc in vio_sig['bodies']:
                            pc.set_color(clist[i])
                        for partname in ('cbars','cmins','cmaxes'):
                            vio_sig[partname].set_edgecolor(clist[i])
                    if not len(data_nonsig[j]) == 0:
                        vio_nonsig = ax[j,2*i1+i2].violinplot(data_nonsig[j], positions = [i-0.1+lg*0.4])
                        # Color of violin plots
                        for pc in vio_nonsig['bodies']:
                            pc.set_color('tab:gray')
                        for partname in ('cbars','cmins','cmaxes'):
                            vio_nonsig[partname].set_edgecolor('tab:gray')
                        
                    # Scatter plots
                    scat_sig = ax[j,2*i1+i2].scatter((i-0.3+lg*0.4)*np.ones(len(data_sig[j])), data_sig[j], c='k')
                    scat_nonsig = ax[j,2*i1+i2].scatter((i-0.1+lg*0.4)*np.ones(len(data_nonsig[j])), data_nonsig[j], 
                                              c='tab:gray')
                    
                
                    if i1 == 0 and i2 == 0:
                        ax[j,0].set_ylabel(["All variables", "Pairs", "Theory"][j], fontsize = 14)
                    # ax[j,2*i1+i2].text(-0.27+i,0.94, repr(len(data_sig))+"         " \
                    #          +repr(len(data_nonsig)), fontsize = 14)
                    ax[j,2*i1+i2].plot(np.arange(-1,6), np.zeros((7)), c='k')
                    # if j == 2:
                    #     ax[j,2*i1+i2].set_xticks(np.arange(len(index_selection)), link_names, fontsize=14)
                    ax[j,2*i1+i2].set_yticks(np.arange(-1.5,1.6,0.5), np.arange(-1.5,1.6,0.5)) #, fontsize=14)
                    ax[j,2*i1+i2].set_xlim([-0.5, len(index_selection)-0.5])
                    ax[j,2*i1+i2].grid(True)
                    
# plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_comparemethods_incllag.pdf")
plt.show()

#%% COMPARE USED DATA ALL V THEORY: ARTICLE
clist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink']
col_list = clist
var_list = ["SPG", "SSS", "MLD", "SubT", "Rho"]
# var_list = ["SPG", "SSS", "MLD", "SubT"]
# col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']


# PCMCI+, RobustParCorr
i1 = 1
i2 = 1

fig, ax = plt.subplots(2,len(index_selection), figsize=(20,9))
for j in range(len(index_selection)):
    for k in range(2):
        # Set lines
        ax[k,j].hlines(5,-1,max_lag+1, color='k', linestyle='-')
        ax[k,j].hlines(10,-1,max_lag+1, color='k', linestyle=':')
    
    # Links including all 5 variables
    # ax[0,j].plot(np.arange(-0.15,max_lag+0.5), links_r[0][i1,i2,j,j],'o', 
    #              c=clist[0], ms=10)
    # ax[0,j].plot(np.arange(-0.15,max_lag+0.5), links_l[0][i1,i2,j,j],'v', 
    #              c=clist[0], ms=10)
    # ax[0,j].plot(np.arange(-0.15,max_lag+0.5), links_c[0][i1,i2,j,j],'X', 
    #              c=clist[0], ms=10)
    # ax[0,j].plot(np.arange(-0.15,max_lag+0.5), links_u[0][i1,i2,j,j],'P', 
    #              c=clist[0], ms=10)
    
    # ax[1,j].plot(np.arange(-0.15,max_lag+0.5), 
    #              links_r[0][i1,i2,j,(j+1)%len(index_selection)],'o', 
    #              c=clist[0], ms=10)
    # ax[1,j].plot(np.arange(-0.15,max_lag+0.5), 
    #              links_l[0][i1,i2,j,(j+1)%len(index_selection)],'v', 
    #              c=clist[0], ms=10)
    # ax[1,j].plot(np.arange(-0.15,max_lag+0.5), 
    #              links_c[0][i1,i2,j,(j+1)%len(index_selection)],'X', 
    #              c=clist[0], ms=10)
    # ax[1,j].plot(np.arange(-0.15,max_lag+0.5), 
    #              links_u[0][i1,i2,j,(j+1)%len(index_selection)],'P', 
    #              c=clist[0], ms=10)
    # ax[1,j].plot(np.arange(-0.15,max_lag+0.5)[0], 
    #              links_r[0][i1,i2,j,(j+1)%len(index_selection),0] \
    #                  + links_u[0][i1,i2,j,(j+1)%len(index_selection),0] \
    #                  + links_c[0][i1,i2,j,(j+1)%len(index_selection),0],'s', 
    #               c=clist[0], ms=10)
   
    # Links for theoretical links
    ax[0,(j+1)%len(index_selection)].plot(np.arange(0.,max_lag+0.5), 
                                          links_r[2][i1,i2,j,2,2],'o', 
                                          c=clist[0], ms=10)
    # ax[0,j].plot(np.arange(0.,max_lag+0.5), links_r[2][i1,i2,j,1,1],'o', 
    #              c=clist[0], ms=10)
    # ax[0,j].plot(np.arange(0.,max_lag+0.5), links_l[2][i1,i2,j,1,1],'v', 
    #              c=clist[1], ms=10)
    # ax[0,j].plot(np.arange(0.,max_lag+0.5), links_c[2][i1,i2,j,1,1],'X', 
    #              c=clist[2], ms=10)
    # ax[0,j].plot(np.arange(0.,max_lag+0.5), links_u[2][i1,i2,j,1,1],'P', 
    #              c=clist[3], ms=10)

    ax[1,j].plot(np.arange(0.,max_lag+0.5), links_r[2][i1,i2,j,1,2],'o', 
                 c=clist[0], ms=10)
    ax[1,j].plot(np.arange(0.,max_lag+0.5)[0], links_l[2][i1,i2,j,1,2,0],'v', 
                  c=clist[1], ms=10)
    ax[1,j].plot(np.arange(0.,max_lag+0.5)[0], links_c[2][i1,i2,j,1,2,0],'X', 
                  c=clist[2], ms=10)
    ax[1,j].plot(np.arange(0.,max_lag+0.5)[0], links_u[2][i1,i2,j,1,2,0],'P', 
                  c=clist[3], ms=10)
    ax[1,j].plot(np.arange(0.,max_lag+0.5)[0], 
                 links_r[2][i1,i2,j,1,2,0] + links_c[2][i1,i2,j,1,2,0] \
                     + links_u[2][i1,i2,j,1,2,0],'s', c=clist[4], ms=10)
        
    # Plot settings
    for k in range(2):
        ax[k,j].set_xlim([-1,max_lag+1])
        ax[k,j].set_ylim([0,len(mod_list)+1])
        ax[k,j].tick_params(labelsize=22)
        ax[k,0].set_ylabel("Number of Models", fontsize=24)
        ax[k,j].grid(True)
    
    ax[1,j].set_xlabel("Lag (years)", fontsize=24)
    ax[0,j].set_title(var_list[j]+r"$\rightarrow$"+var_list[j], fontsize=24)
    ax[1,j].set_title(var_list[j]+r"$\rightarrow$"+var_list[(j+1)%len(var_list)], 
                      fontsize=24)
    
    # Change color of axes
    plt.setp(ax[1,j].spines.values(), color=col_list[j], linewidth=4)
    plt.setp([ax[1,j].get_xticklines(), ax[0,j].get_yticklines()], color=col_list[j])
    
plt.tight_layout()
# plt.savefig(dir_save_fig + "NrModelsWithLink_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_paper.pdf")
# plt.savefig(dir_save_fig + "NrModelsWithLink_norho_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_paper.pdf")
plt.show()

#%% CHECK WHICH MODELS

lnk = 0

for lag in [0,1]:
    print("Lag: "+repr(lag))
    print(len(np.array(mod_list)[where_right[2][i1,i2,:,lnk,1,2,lag]==1]))
    print(np.array(mod_list)[where_right[2][i1,i2,:,lnk,1,2,lag]==1])
    print(len(np.array(mod_list)[where_contemp[2][i1,i2,:,lnk,1,2,lag]==1]))
    print(np.array(mod_list)[where_contemp[2][i1,i2,:,lnk,1,2,lag]==1])
    print(len(np.array(mod_list)[where_unclear[2][i1,i2,:,lnk,1,2,lag]==1]))
    print(np.array(mod_list)[where_unclear[2][i1,i2,:,lnk,1,2,lag]==1])
    # print(len(np.array(mod_list)[where_left[2][i1,i2,:,lnk,1,2,lag]==1]))
    # print(np.array(mod_list)[where_left[2][i1,i2,:,lnk,1,2,lag]==1])

#%%

# Get the models with a significant link
where_check = [sum(x) for x in zip(where_right, where_contemp, 
                                   where_unclear)]
# Get the models with a significant link
where_uc = [sum(x) for x in zip(where_contemp, where_unclear)]

where_link = where_check[2][i1,i2,:,:,1,2,0] + where_check[2][i1,i2,:,:,1,2,1] + where_check[2][i1,i2,:,:,1,2,2]
where_link[where_link == 2] = 1
where_link[where_link == 3] = 1

print(np.sum(where_check[2][i1,i2,:,:,1,2,0], axis=1))
print(np.sum(where_check[2][i1,i2,:,:,1,2,1], axis=1))
print(np.sum(where_check[2][i1,i2,:,:,1,2,0:2], axis=(1,2)))
print(np.sum(where_uc[2][i1,i2,:,:,1,2,0], axis=1))
print(np.sum(where_link, axis=1))

print(np.sum(where_check[2][i1,i2,:,:,1,2,0], axis=0))
print(np.sum(where_check[2][i1,i2,:,:,1,2,1], axis=0))
print(np.sum(where_check[2][i1,i2,:,:,1,2,2], axis=0))
print(np.sum(where_check[2][i1,i2,:,:,1,2,0:3], axis=(0,2)))
print(np.sum(where_right[2][i1,i2,:,:,1,2,0], axis=0))
print(np.sum(where_uc[2][i1,i2,:,:,1,2,0], axis=0))
print(np.sum(where_link, axis=0))

#%% VIOLIN PLOT: ARTICLE

link_names = ["                Lag-1 \nSPG to SSS", 
              "Lag-0       Lag-1 \nSSS to MLD", 
              "Lag-0       Lag-1 \nMLD to SubT", 
              "Lag-0       Lag-1 \nSubT to Rho", 
              "Lag-0       Lag-1 \nRho to SPG"]
col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# link_names = ["                Lag-1 \nSPG to SSS", 
#               "Lag-0       Lag-1 \nSSS to MLD", 
#               "Lag-0       Lag-1 \nMLD to SubT", 
#               "Lag-1       Lag-2 \nSubT to SPG"]
# col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

alpha = 0.4

# face_list = [[] for i in range(len(col_list))]
# edge_list = [[] for i in range(len(col_list))]
# for i in range(len(col_list)):
#     face_list[i] = mpl.colors.to_rgb(col_list[i]) + (alpha,)
#     edge_list[i] = mpl.colors.to_rgb(col_list[i]) + (0.99,)

# PCMCI+, RobustParCorr
i1 = 1
i2 = 1
    
# lag_list = [[1], [0,1], [0], [0,1], [1]]
lag_list = [[1], [0,1], [0,1], [0,1], [0,1]]
# lag_list = [[1], [1], [0,1], [1], [1]]
# lag_list = [[1], [0,1], [0,1], [1,2]]

fig = plt.figure(figsize=(15,8))

for i in range(len(index_selection)):
    
    # Get the models with a significant link
    where_check = [sum(x) for x in zip(where_right, where_contemp, 
                                        where_unclear)]
    # where_check = where_right
    
    # For each of the lags
    for lag_ind in range(len(lag_list[i])):
        # Get causal effect and lag
        lg = lag_list[i][lag_ind]
        est_ce = estimated_causal_effects_list[:,:,lag_ind]
        
        # Theory
        data_sig = est_ce[where_check[2][i1,i2,:,i,1,2,lg] == 1,i]
        data_nonsig = est_ce[where_check[2][i1,i2,:,i,1,2,lg] == 0,i]
        
        print(np.array(mod_list)[where_check[2][i1,i2,:,i,1,2,lg] == 1])
        
        if lag_list[i][-1] == 2:
            lg = lg - 1
            
        # Violin plots
        vio_sig = plt.violinplot(data_sig, positions = [2*i+lg-0.2])
        vio_nonsig = plt.violinplot(data_nonsig, positions = [2*i+lg+0.2])
        # vio_all = plt.violinplot(est_ce[:,i], positions = [2*i+lg-0.2])
        
        # Color of violin plots
        for pc in vio_sig['bodies']:
            pc.set_color(col_list[i])
            pc.set_linewidth(2)
            pc.set_alpha(0.4)
            if lg == 1:
                pc.set_hatch('xx')
        for pc in vio_nonsig['bodies']:
            pc.set_color('tab:gray')
            pc.set_linewidth(2)
            pc.set_alpha(0.4)
            if lg == 1:
                htch = pc.set_hatch('xx')
        # for pc in vio_all['bodies']:
        #     pc.set_facecolor('white')
        #     pc.set_edgecolor(col_list[i])
        #     pc.set_linewidth(2)
        #     pc.set_linestyle(':')
                
        for partname in ('cbars','cmins','cmaxes'):
            vio_sig[partname].set_edgecolor(col_list[i])
            vio_sig[partname].set_linewidth(2)
            vio_nonsig[partname].set_edgecolor('tab:gray')
            vio_nonsig[partname].set_linewidth(2)
            
        # Scatter plots
        scat_sig = plt.scatter((2*i+lg-0.2)*np.ones(len(data_sig)), 
                               data_sig, c='k')
        scat_nonsig = plt.scatter((2*i+lg+0.2)*np.ones(len(data_nonsig)), 
                                  data_nonsig, c='tab:gray')
        
        plt.text(-0.3+2*i+lg,1.25, repr(len(data_sig))+"   " \
                 +repr(len(data_nonsig)), fontsize = 18)
            
plt.plot(np.arange(-1,12), np.zeros((13)), c='k')
plt.xlim([0, 2*len(index_selection)-0.5])
# plt.xlim([-0.5, 2*len(index_selection)-0.5])
plt.ylim([-1.,1.2])
plt.xticks(np.arange(0.5,2*len(index_selection),2), link_names, fontsize=20)
plt.yticks(fontsize=18)
plt.ylabel("Causal Effect", fontsize=20)
plt.grid()
plt.tight_layout()

# plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_paper_new.png", dpi=300)
# plt.savefig(dir_save_fig + "Causaleffect_norho_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_new.png", dpi=300)
plt.show()

#%% SPG TO ITSELF LAG
"""
CHECK!!
"""

# col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# link_names = ["Lag-1 \nSSS", 
#               "Lag-1 \nMLD", 
#               "Lag-1       Lag-2 \nSubT", 
#               "Lag-1 \nRho",
#               "Lag-1       Lag-2       Lag-3       Lag-4       Lag-5 \nSPG"]
col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
link_names = ["Lag-1 \nSSS", 
              "Lag-1 \nMLD", 
              "Lag-1       Lag-2 \nSubT",
              "Lag-1       Lag-2 \nSPG"]

alpha = 0.4

# face_list = [[] for i in range(len(col_list))]
# edge_list = [[] for i in range(len(col_list))]
# for i in range(len(col_list)):
#     face_list[i] = mpl.colors.to_rgb(col_list[i]) + (alpha,)
#     edge_list[i] = mpl.colors.to_rgb(col_list[i]) + (0.99,)

# PCMCI+, RobustParCorr
i1 = 1
i2 = 1
# i=0
    
# lag_list = [[1], [0,1], [0], [0,1], [1]]
# lag_list = [[1], [1], [1,2], [1], [1,2,3,4,5]]
# lag_list = [[1], [1], [0,1], [1], [1]]
lag_list = [[1,2], [1], [1,2], [1,2,3,4,5]]

# posit_list = [[0], [1], [2,3], [4], [5,6,7,8,9]]
posit_list = [[0,1], [2], [3,4], [5,6,7,8,9]]

fig = plt.figure(figsize=(15,8))
# Get the models with a significant link
where_check = [sum(x) for x in zip(where_right, where_contemp, 
                                    where_unclear)]
# where_check = where_right

# For each of the lags
# for i in range(len(index_selection)):
    # lag_ind = 1
for i in range(len(index_selection)):
    for lag_ind in range(len(lag_list[i])):
        # Get causal effect and lag
        lg = lag_list[i][lag_ind]
        est_ce = estimated_causal_effects_list[:,:,lag_ind]
        
        # Theory
        # data_sig = est_ce[where_check[2][i1,i2,:,i,1,1,lg] == 1,i]
        # data_nonsig = est_ce[where_check[2][i1,i2,:,i,1,1,lg] == 0,i]
        # print(np.array(mod_list)[where_check[2][i1,i2,:,i,1,1,lg] == 1])
        
        # Violin plots
        # vio_sig = plt.violinplot(data_sig, positions = [posit_list[i][lag_ind]-0.2])
        # vio_nonsig = plt.violinplot(data_nonsig, positions = [posit_list[i][lag_ind]+0.2])
        # vio_all = plt.violinplot(est_ce[:,i], positions = [2*i+lg-0.2])
        
        data_sig = est_ce[where_check[2][i1,i2,:,i,2,2,lg] == 1,i]
        vio_sig = plt.violinplot(data_sig, positions = [posit_list[i][lag_ind]-0.2])
        scat_sig = plt.scatter((posit_list[i][lag_ind]-0.2)*np.ones(len(data_sig)), 
                               data_sig, c='k')
        if not (i == 2 and lag_ind == 0):
            data_nonsig = est_ce[where_check[2][i1,i2,:,i,2,2,lg] == 0,i]
            vio_nonsig = plt.violinplot(data_nonsig, positions = [posit_list[i][lag_ind]+0.2])
            scat_nonsig = plt.scatter((posit_list[i][lag_ind]+0.2)*np.ones(len(data_nonsig)), 
                                      data_nonsig, c='tab:gray')
        else:
            data_nonsig = []
        
        print(np.array(mod_list)[where_check[2][i1,i2,:,i,2,2,lg] == 1])
        
        # if lag_list[i][-1] == 2:
        #     lg = lg - 1
            
        
        # Color of violin plots
        for pc in vio_sig['bodies']:
            pc.set_color(col_list[i])
            pc.set_linewidth(2)
            pc.set_alpha(0.4)
            if lg == 2:
                pc.set_hatch('xx')
            if lg == 3:
                pc.set_hatch('//')
            if lg == 4:
                pc.set_hatch('--')
            if lg == 5:
                pc.set_hatch('++')
        for pc in vio_nonsig['bodies']:
            pc.set_color('tab:gray')
            pc.set_linewidth(2)
            pc.set_alpha(0.4)
            if lg == 2:
                pc.set_hatch('xx')
            if lg == 3:
                pc.set_hatch('//')
            if lg == 4:
                pc.set_hatch('--')
            if lg == 5:
                pc.set_hatch('++')
        # for pc in vio_all['bodies']:
        #     pc.set_facecolor('white')
        #     pc.set_edgecolor(col_list[i])
        #     pc.set_linewidth(2)
        #     pc.set_linestyle(':')
                
        for partname in ('cbars','cmins','cmaxes'):
            vio_sig[partname].set_edgecolor(col_list[i])
            vio_sig[partname].set_linewidth(2)
            vio_nonsig[partname].set_edgecolor('tab:gray')
            vio_nonsig[partname].set_linewidth(2)
            
        # Scatter plots
        # scat_sig = plt.scatter((posit_list[i][lag_ind]-0.2)*np.ones(len(data_sig)), 
        #                        data_sig, c='k')
        # scat_nonsig = plt.scatter((posit_list[i][lag_ind]+0.2)*np.ones(len(data_nonsig)), 
        #                           data_nonsig, c='tab:gray')
        
        plt.text(-0.3+posit_list[i][lag_ind],1.25, repr(len(data_sig))+"   " \
                 +repr(len(data_nonsig)), fontsize = 18)
            
plt.plot(np.arange(-1,14), np.zeros((15)), c='k')
# plt.xlim([-0.5, 2*len(index_selection)-0.5])
plt.ylim([-1.,1.2])
plt.xlim([-0.5, 9.5])
plt.xticks(np.array([0.5,2,3.5,7]), link_names, fontsize=20)
# plt.xlim([-0.5, 5.5])
# plt.xticks(np.array([0,1,2.5,4.5]), link_names, fontsize=20)
plt.yticks(fontsize=18)
plt.ylabel("Causal Effect", fontsize=20)
plt.grid()
plt.tight_layout()

# plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_auto-effect_paper.png", dpi=300)
# plt.savefig(dir_save_fig + "Causaleffect_norho_alpha"+repr(alpha_lev)+"_"
#             +meth_list[i1]+"_"+metric_list[i2]+"_auto-effect.png", dpi=300)
plt.show()

#%% MODEL VALUES: ARTICLE SI

link_names = ["                Lag-1 \nSPG to SSS", 
              "Lag-0       Lag-1 \nSSS to MLD", 
              "Lag-0       Lag-1 \nMLD to SubT", 
              "Lag-0       Lag-1 \nSubT to Rho",
              "Lag-0       Lag-1 \nRho to SPG"]
col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

col_modlist = ['tab:blue','tab:orange','tab:orange','tab:green','tab:green',
                'tab:green','tab:green','tab:red','tab:red','tab:purple',
                'tab:purple','tab:brown','tab:brown','tab:brown','tab:brown',
                'tab:brown','tab:pink','tab:pink','tab:gray','tab:gray',
                'tab:olive','tab:olive','gold','chocolate','tab:cyan','tab:cyan',
                'tab:cyan','black','lightgreen','lightgreen','navy','navy']
mark_modlist = ['o','o','X','o','X','v','P','o','X','o','X','o','X','v','P',
                'd','o','X','o','X','o','X','o','o','o','X','v','o','o','X',
                'o','X']
lag_list = [[1], [0,1], [0,1], [0,1], [0,1]]

# link_names = ["                Lag-1 \nSPG to SSS", 
#               "Lag-0       Lag-1 \nSSS to MLD", 
#               "Lag-0       Lag-1 \nMLD to SubT", 
#               "Lag-1       Lag-2 \nSubT to SPG"]
# col_modlist = ['tab:blue','tab:orange','tab:orange','tab:green','tab:green',
#                 'tab:green','tab:green','tab:red','tab:red','tab:purple',
#                 'tab:purple','tab:brown','tab:brown','tab:brown','tab:brown',
#                 'tab:brown','tab:brown','tab:pink','tab:pink','tab:gray','tab:gray',
#                 'tab:olive','tab:olive','gold','chocolate','tab:cyan','tab:cyan',
#                 'tab:cyan','black','lightgreen','lightgreen','navy','navy']
# mark_modlist = ['o','o','X','o','X','v','P','o','X','o','X','o','X','v','P',
#                 'd','*','o','X','o','X','o','X','o','o','o','X','v','o','o','X',
#                 'o','X']
# lag_list = [[1], [0,1], [0,1], [1,2]]

# PCMCI+, RobustParCorr
i1 = 1
i2 = 1

col_modsig = np.array(col_modlist.copy())

# Get color when significant
sig_list = np.array([where_check[2][i1,i2,:,i,1,2,:] 
                      for i in range(len(index_selection))])
# allsig = np.array([all(sig_list[:,m] < p_thres) for m in range(len(mod_list))])
# col_modsig[allsig==False] = None

# Based on data being significant when not correcting for confounding factors
fig = plt.figure(figsize=(17,11))

# Scatter plots
for m in range(len(mod_list)):
    for i in range(len(index_selection)):
        
        # For each of the lags
        for lag_ind in range(len(lag_list[i])):
            # Get causal effect and lag
            est_ce = estimated_causal_effects_list[:,:,lag_ind]
            lg = lag_list[i][lag_ind]
            
            if lag_list[i][-1] == 2:
                lg = lg - 1
            
            if lg == 0:
                col_face = np.array(col_modlist.copy())
            elif lg == 1:
                col_face = len(mod_list) * ['white']
            
            if i == 1 and lg == 0:
                plt.scatter(np.arange(2*i+lg-0.3,2*i+lg+0.299,0.6/len(mod_list))[m], 
                            est_ce[m,i], marker=mark_modlist[m],
                            edgecolors=col_modlist[m], facecolors=col_face[m],
                            label=mod_list[m], s=120)
            else:
                plt.scatter(np.arange(2*i+lg-0.3,2*i+lg+0.299,0.6/len(mod_list))[m], 
                            est_ce[m,i], marker=mark_modlist[m],
                            edgecolors=col_modlist[m], facecolors=col_face[m],
                            s=120)

plt.plot(np.arange(-1,12), np.zeros((13)), c='k')

plt.xlim([0, 2*len(index_selection)-0.5])
plt.ylim([-1.,1.2])
plt.ylabel("Causal Effect", fontsize=20)
plt.xticks(np.arange(0.5,2*len(index_selection),2), link_names, fontsize=20)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(fontsize=14, ncols=4, loc='upper right', bbox_to_anchor=(0.995, 1.34))
plt.tight_layout()
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha"+repr(alpha_lev)+"_alltheory.pdf")
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha"+repr(alpha_lev)+"_inclparent.pdf")
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha.pdf")
# plt.savefig(dir_save_fig + "Causaleffect_modelscatter_alpha"+repr(alpha_lev)+
#             "_"+meth_list[i1]+"_"+metric_list[i2]+"_paper.png", dpi=300)
# plt.savefig(dir_save_fig + "Causaleffect_norho_modelscatter_alpha"+repr(alpha_lev)+
#             "_"+meth_list[i1]+"_"+metric_list[i2]+"_paper_new.png", dpi=300)
plt.show()

#%%

i0 = 0

for i1 in range(2):
    for i2 in range(2):
        # Based on data being significant when not correcting for confounding factors
        fig = plt.figure(figsize=(12,8))
        
        for i in range(len(index_selection)):
            # Split data in significant and non-significant
            # data_sig = estimated_causal_effects[parcorr_theory[:,i,1,1,2,0] < p_thres,i]
            # data_nonsig = estimated_causal_effects[parcorr_theory[:,i,1,1,2,0] > p_thres,i]
            
            # All 5 variables
            # data_sig = estimated_causal_effects[parcorr_all[0][i1,i2,:,1,i,(i+1)%len(index_selection),0] < p_thres,i]
            # if not i == 1:
            #     data_nonsig = estimated_causal_effects[parcorr_all[0][i1,i2,:,1,i,(i+1)%len(index_selection),0] > p_thres,i]
            
            # Theory
            data_sig = estimated_causal_effects[parcorr_all[2][i1,i2,:,i,1,1,2,0] < p_thres,i]
            data_nonsig = estimated_causal_effects[parcorr_all[2][i1,i2,:,i,1,1,2,0] > p_thres,i]
            
            # Violin plots
            vio_sig = plt.violinplot(data_sig, positions = [i-0.2])
            vio_nonsig = plt.violinplot(data_nonsig, positions = [i+0.2])
            
            # Scatter plots
            scat_sig = plt.scatter((i-0.2)*np.ones(len(data_sig)), data_sig, c='k')
            scat_nonsig = plt.scatter((i+0.2)*np.ones(len(data_nonsig)), data_nonsig, 
                                      c='tab:gray')
            
            # Color of violin plots
            for pc in vio_sig['bodies']:
                pc.set_color(col_list[i])
            for pc in vio_nonsig['bodies']:
                pc.set_color('tab:gray')
            for partname in ('cbars','cmins','cmaxes'):
                vio_sig[partname].set_edgecolor(col_list[i])
                vio_nonsig[partname].set_edgecolor('tab:gray')
        
            plt.text(-0.27+i,0.94, repr(len(data_sig))+"         " \
                     +repr(len(data_nonsig)), fontsize = 14)
        
        plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
        plt.xticks(np.arange(len(index_selection)), link_names, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([-0.5, len(index_selection)-0.5])
        plt.grid()
        # plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_inclparent.pdf")
        plt.show()

#%%
# Based on data being significant when including all factors
fig = plt.figure(figsize=(12,8))

for i in range(len(index_selection)):
    # Split data in significant and non-significant
    data_sig = estimated_causal_effects[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),0] < p_thres,i]
    data_nonsig = estimated_causal_effects[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),0] > p_thres,i]
    
    # Violin plots
    vio_sig = plt.violinplot(data_sig, positions = [i-0.2])
    if len(data_nonsig) > 0:
        vio_nonsig = plt.violinplot(data_nonsig, positions = [i+0.2])
    
    # Scatter plots
    scat_sig = plt.scatter((i-0.2)*np.ones(len(data_sig)), data_sig, c='k')
    scat_nonsig = plt.scatter((i+0.2)*np.ones(len(data_nonsig)), data_nonsig, 
                              c='tab:gray')
    
    # Color of violin plots
    for pc in vio_sig['bodies']:
        pc.set_color(col_list[i])
    for pc in vio_nonsig['bodies']:
        pc.set_color('tab:gray')
    for partname in ('cbars','cmins','cmaxes'):
        vio_sig[partname].set_edgecolor(col_list[i])
        vio_nonsig[partname].set_edgecolor('tab:gray')

    plt.text(-0.27+i,0.94, repr(len(data_sig))+"         " \
             +repr(len(data_nonsig)), fontsize = 14)

plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
plt.xticks(np.arange(len(index_selection)), link_names, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([-0.5, len(index_selection)-0.5])
plt.grid()
# plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_allsig.pdf")
plt.show()

#%%

# sig_list = np.array([parcorr[:,1,i,(i+1)%len(index_selection),0] 
#                      for i in range(len(index_selection))])
# allsig = np.array([all(sig_list[:,m] < p_thres) for m in range(len(mod_list))])
for i1 in range(2):
    for i2 in range(2):
        sig_list = np.array([parcorr_all[2][i1,i2,:,i,1,1,2,0] 
                              for i in range(len(index_selection))])
        # sig_list = np.array([parcorr_theory_gpdc[:,i,1,1,2,0] 
        #                       for i in range(len(index_selection))])
        allsig = np.array([all(sig_list[:,m] < p_thres) for m in range(len(mod_list))])
        
        # Based on data being significant when including all factors
        fig = plt.figure(figsize=(12,8))
        
        for i in range(len(index_selection)):
            # Split data in significant and non-significant
            data_sig = estimated_causal_effects[allsig == True,i]
            data_nonsig = estimated_causal_effects[allsig == False,i]
            
            # Violin plots
            vio_sig = plt.violinplot(data_sig, positions = [i-0.2])
            if len(data_nonsig) > 0:
                vio_nonsig = plt.violinplot(data_nonsig, positions = [i+0.2])
            
            # Scatter plots
            scat_sig = plt.scatter((i-0.2)*np.ones(len(data_sig)), data_sig, c='k')
            scat_nonsig = plt.scatter((i+0.2)*np.ones(len(data_nonsig)), data_nonsig, 
                                      c='tab:gray')
            
            # Color of violin plots
            for pc in vio_sig['bodies']:
                pc.set_color(col_list[i])
            for pc in vio_nonsig['bodies']:
                pc.set_color('tab:gray')
            for partname in ('cbars','cmins','cmaxes'):
                vio_sig[partname].set_edgecolor(col_list[i])
                vio_nonsig[partname].set_edgecolor('tab:gray')
        
            plt.text(-0.27+i,0.94, repr(len(data_sig))+"         " \
                     +repr(len(data_nonsig)), fontsize = 14)
        
        plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
        plt.xticks(np.arange(len(index_selection)), link_names, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([-0.5, len(index_selection)-0.5])
        plt.grid()
        # plt.savefig(dir_save_fig + "Causaleffect_alpha"+repr(alpha_lev)+"_alltheory.pdf")
        plt.show()

#%% Which models

modlist = np.array(mod_list)

modlist_theory = [[[],[]] for i in range(len(index_selection))]
modlist_sig = [[[],[]] for i in range(len(index_selection))]
modlist_allsig = [[],[]]

for i in range(len(index_selection)):
    modlist_theory[i][0] = modlist[parcorr_theory[:,i,1,1,2,0] < p_thres]
    modlist_theory[i][1] = modlist[parcorr_theory[:,i,1,1,2,0] > p_thres]
    
    modlist_sig[i][0] = modlist[parcorr[:,1,i,(i+1)%len(index_selection),0] < p_thres]
    modlist_sig[i][1] = modlist[parcorr[:,1,i,(i+1)%len(index_selection),0] > p_thres]

modlist_allsig[0] = modlist[allsig == True]
modlist_allsig[1] = modlist[allsig == False]

for i in range(len(index_selection)):
    print(link_names[i])
    for j in range(2):
        print(["Link", "No link"][j])
        print(modlist_theory[i][j])
        print(modlist_sig[i][j])

print("All")
for j in range(2):
    print(["Link", "No link"][j])
    print(modlist_allsig[j])
    for i in range(len(index_selection)):
        # Split data in significant and non-significant
        data_sig = estimated_causal_effects[ \
                        allsig == True,i]
        data_nonsig = estimated_causal_effects[ \
                        allsig == False,i]

#%%

# sig_list = np.array([parcorr[:,i,1,1,2,0] 
#                       for i in range(len(index_selection))])
# allsig = np.array([all(sig_list[:,m] < p_thres) for m in range(len(mod_list))])

# Based on data being significant when including all factors
fig = plt.figure(figsize=(24,8))

for i in range(len(index_selection)):
    # Split data in significant and non-significant
    data_dir_sig = estimated_causal_effects_lag[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),0] < p_thres,i,0]
    data_dir_nonsig = estimated_causal_effects_lag[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),0] > p_thres,i,0]
    data_lag_sig = estimated_causal_effects_lag[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),1] < p_thres,i,1]
    data_lag_nonsig = estimated_causal_effects_lag[ \
                    parcorr[:,1,i,(i+1)%len(index_selection),1] > p_thres,i,1]
    
    
    # Violin plots
    vio_dir_sig = plt.violinplot(data_dir_sig, positions = [i-0.2])
    if len(data_dir_nonsig) > 0:
        vio_dir_nonsig = plt.violinplot(data_dir_nonsig, positions = [i+0.2])
    vio_lag_sig = plt.violinplot(data_lag_sig, 
                                 positions = [len(index_selection)+i-0.2])
    vio_lag_nonsig = plt.violinplot(data_lag_nonsig, 
                                    positions = [len(index_selection)+i+0.2])
    
    # Scatter plots
    scat_dir_sig = plt.scatter((i-0.2)*np.ones(len(data_dir_sig)), data_dir_sig, 
                               c='k')
    scat_dir_nonsig = plt.scatter((i+0.2)*np.ones(len(data_dir_nonsig)), 
                                  data_dir_nonsig, c='tab:gray')
    scat_lag_sig = plt.scatter((len(index_selection)+i-0.2)*np.ones(len(data_lag_sig)), 
                               data_lag_sig, c='k')
    scat_lag_nonsig = plt.scatter((len(index_selection)+i+0.2)*np.ones(len(data_lag_nonsig)), 
                                  data_lag_nonsig, c='tab:gray')
     
    # Color of violin plots
    for pc in vio_dir_sig['bodies']:
        pc.set_color(col_list[i])
    for pc in vio_dir_nonsig['bodies']:
        pc.set_color('tab:gray')
    for pc in vio_lag_sig['bodies']:
        pc.set_color(col_list[i])
    for pc in vio_lag_nonsig['bodies']:
        pc.set_color('tab:gray')
    for partname in ('cbars','cmins','cmaxes'):
        vio_dir_sig[partname].set_edgecolor(col_list[i])
        vio_dir_nonsig[partname].set_edgecolor('tab:gray')
        vio_lag_sig[partname].set_edgecolor(col_list[i])
        vio_lag_nonsig[partname].set_edgecolor('tab:gray')
 
     # plt.text(-0.27+i,0.94, repr(len(data_sig))+"         " \
     #          +repr(len(data_nonsig)), fontsize = 14)

plt.plot(np.arange(-1,12), np.zeros((13)), c='k')
plt.xticks(np.arange(2*len(index_selection)), 
           np.array([link_names, link_names]).flatten(), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([-0.5, 2*len(index_selection)-0.5])
plt.grid()
# plt.savefig(dir_save_fig + "Causaleffect_lag1_alpha"+repr(alpha_lev)+".pdf")
plt.show()


#%% Load Data gpdc

# # Initialise correlations, partial correlations and graphs
# corr_theory_gpdc = np.empty((len(mod_list),len(index_selection),2,3,3,max_lag+1))
# parcorr_theory_gpdc = np.empty((len(mod_list),len(index_selection),2,3,3,max_lag+1))
# graph_list_theory_gpdc = [[[] for i in range(len(index_selection))] 
#               for m in range(len(mod_list))]

# # For each model
# for m in range(len(mod_list)):
#     res_gpdc = np.load(dir_save_data+"/gpdc_models/causaldiscovery_gpdc_theory_"
#                        +mod_list[m]+".npz")
#     corr_theory_gpdc[m] = res_gpdc['corr']
#     parcorr_theory_gpdc[m] = res_gpdc['parcorr']
#     graph_list_theory_gpdc[m] = res_gpdc['graph']

# # To array
# graph_theory_gpdc = np.array(graph_list_theory_gpdc)

#%% MODELS WITH LINKS
# Identify where each model has a link (1)
# where_right = np.where(graph == '-->',1,0)
# where_contemp = np.where(graph[:,:,:,0] == 'o-o',1,0)

# where_2p_right = np.where(graph_2p == '-->',1,0)
# where_2p_contemp = np.where(graph_2p[:,:,:,:,:,0] == 'o-o',1,0)

# where_theory_right = np.where(graph_theory[:,:,1::,1::] == '-->',1,0)
# where_theory_contemp = np.where(graph_theory[:,:,1::,1:,0] == 'o-o',1,0)

# # where_theory_gpdc_right = np.where(graph_theory_gpdc[:,:,1::,1::] == '-->',1,0)
# # where_theory_gpdc_contemp = np.where(graph_theory_gpdc[:,:,1::,1:,0] == 'o-o',1,0)

# # Sum to obtain to number of models with a link
# links = np.sum(where_right, axis=0)
# links[:,:,0] = np.sum(where_contemp, axis=0)

# links_2p = np.sum(where_2p_right, axis=0)
# links_2p[:,:,:,:,0] = np.sum(where_2p_contemp, axis=0)

# links_theory = np.sum(where_theory_right, axis=0)
# links_theory[:,:,:,0] = np.sum(where_theory_contemp, axis=0)

# links_theory_gpdc = np.sum(where_theory_gpdc_right, axis=0)
# links_theory_gpdc[:,:,:,0] = np.sum(where_theory_gpdc_contemp, axis=0)

#%% Scatter plot

# link_names = ["SPG to SSS", "SSS to MLD", "MLD to Sub. Th.", 
#               "Sub. Th. to Density", "Density to SPG"]

# p_thres = alpha_lev


# fig = plt.figure(figsize=(12,8))
# for i in range(len(index_selection)):
#     col_pcorr = np.where(parcorr_theory[:,i,1,0,1,0] < p_thres, 'tab:purple', 'tab:gray')
#     plt.scatter(i * np.ones(len(mod_list)), estimated_causal_effects[:,i],
#                 c=col_pcorr)

# plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
# plt.xticks(np.arange(len(index_selection)), link_names)
# plt.xlim([-0.5, len(index_selection)-0.5])
# plt.grid()
# plt.show()

# fig = plt.figure(figsize=(12,8))
# for i in range(len(index_selection)):
#     col_pcorr = np.where(parcorr[:,0,i,(i+1)%len(index_selection),0] < pval, 'tab:purple', 'tab:gray')
#     plt.scatter(i * np.ones(len(mod_list)), estimated_causal_effects[:,i],
#                 c=col_pcorr)

# plt.plot(np.arange(-1,6), np.zeros((7)), c='k')
# plt.xticks(np.arange(len(index_selection)), link_names)
# plt.xlim([-0.5, len(index_selection)-0.5])
# plt.grid()
# plt.show()

#%% Plot Number of Links

# clist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#          'tab:brown', 'tab:pink']

# fig, ax = plt.subplots(len(index_selection),len(index_selection), 
#                        figsize=(10,10))

# for i in range(len(index_selection)):
#     for j in range(len(index_selection)):
#         ax[i,j].hlines(5,-1,max_lag+1, color='k', linestyle='-')
#         ax[i,j].hlines(10,-1,max_lag+1, color='k', linestyle=':')
#         # Links including all 5 variables
#         ax[i,j].plot(np.arange(0,max_lag+1), links[i,j],'o', c='tab:blue')
#         # Links focussing on pairwise links
#         # if j == i:
#         #     for k in range(len(index_selection)):
#         #         ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,k,0,0],'X', c=clist[k+2], markerfacecolor='none')
#         # elif i < j:
#         #     ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,j,0,1],'X', c='tab:green')
#         # elif i > j:
#         #     ax[i,j].plot(np.arange(0,max_lag+1), links_2p[i,j,1,0],'X', c='tab:green')
#         # Links focussing on theoretical links
#         if j == i:
#             ax[i,j].plot(np.arange(0,max_lag+1), links_theory[i,0,0],'v', c='tab:orange')
#         if j == (i+1)%len(index_selection):
#             ax[i,j].plot(np.arange(0,max_lag+1), links_theory[i,0,1],'v', c='tab:orange')
#         # Links focussing on theoretical links using gpdc
#         # if j == i:
#         #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory_gpdc[i,0,0],'^', c='tab:purple')
#         # if j == (i+1)%len(index_selection):
#         #     ax[i,j].plot(np.arange(0,max_lag+1), links_theory_gpdc[i,0,1],'^', c='tab:purple')
#         # Plot settings
#         ax[i,j].set_xlim([-1,max_lag+1])
#         ax[i,j].set_ylim([0,len(mod_list)+1])
#         ax[i,j].grid()
#         if i == len(index_selection)-1:
#             ax[i,j].set_xlabel("Lag (years)", fontsize=8)
#         if i == 0:
#             ax[i,j].set_title(index_selection[j], fontsize=10)
#         if j == 0:
#             ax[i,j].set_ylabel(index_selection[i], fontsize=10)

# plt.tight_layout()
# # plt.savefig(dir_save_fig + "NrModelsWithLink_alpha"+repr(alpha_lev)+".pdf")
# plt.show()