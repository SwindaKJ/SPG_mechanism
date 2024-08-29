Python files for causal inference and causal effect computation to validate the theoretical mechanism of subpolar gyre variability.

Multi-model mean and model mixed layer depth count computation:
In meanmodel_fields.py there is code to compute the (multi-)model means, as well as code to study the mixed layer depth in more detail, e.g. count where it exceeds 1000m.

Index Computation:
To compute the indices used, run indexcomputation.py, which calls on the the two files in inexcomputationfunctions folder. The wget-files to download the data are required to do this, also for the area-files, for which the CMIP_wgetfiles.py file is used. The index computation happens in three steps. First the data is downloaded, then its preprocessed and the index is computed, and lastly the data is deleted again (to save storage). To check the index-computation went ok, run datacheck.py, which plots the timeseries and outputs their length.

Causal Inference and Effect:
The computation of the causal network and causal effect is done in Causal_discovery_effect.py. This code requires the Tigramite package to be installed and the indices to be computed (from indexcomputation.py). The first part is concerned with the causal discovery, where in the second part the causal effect is computed and plotted.
