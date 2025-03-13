# Notebooks

The point mass balance dataset is found in the Data-folder: 
Data/2023-08-28_stake_mb_Hydra2_corrected.csv

Please use and refer to the original source:
https://doi.org/10.58059/sjse-6w92

## Overview of Notebooks and scripts:

### 0 Processing pipeline
0.1_data_processing.ipynb - Data processing and cleaning. 

0.2_get_glims_rgi_id.ipynb - Get RGIIDs corresponding to Norwegian glacier ids (breid). 

0.3_reproject_coordinates.ipynb - Get lat/lon coordinates corresponding to UTM 32/33/34 coordinates in dataset. 

0.4_get_oggm_data.ipynb - For each point/coordinate, get topographical features using the OGGM pipeline using RGIID. 

0.5_get_climate_data.ipynb - For each point/coordinate/year, get climate variables and altitude of climate data from ERA5-Land. 

0.6.1_compute_skyview.py and 0.6.2_add_skyview_factor.ipynb - Compute and add the skyview factor as a feature.

### 1 Data exploration
1.1_data_exploration.ipynb - Scipt for data exploration. <br>
&emsp; ***Figure 2***

### 2 Model training and predictions
2.1.1_model_training_validation.ipynb - Set up test and training datasets and cross validation. 

2.1.2_model_training_cluster.py - Hyperparameter tuning using cross validation scheme.  

2.2_model_predictions.ipynb - Predict mass Balance in each grid cell for each glacier and month using trained model.

### 3 Analysis and plotting scripts
3.1_ML_model_performance_test.ipynb - Evaluate performance of trained model on test dataset. <br>
&emsp; ***Figure 5 and C1*** 

3.2_prepare_model_comparison.ipynb - Process results from GloGEM, OGGM and PyGEM for model comparison. 

3.3_model_comparison_point.ipynb - Model comparison on point mass balance. <br>
&emsp; ***Figure 6*** 

3.4_model_comparison_glacier_wide.ipynb - Model comparison on annual and seasonal glacier-wide mass balance. <br>
&emsp; ***Figure 8*** 

3.5_model_comparison_monthly.ipynb - Model comparison on monthly glacier-wide mass balance. <br>
&emsp; ***Figure 9*** 

3.6_model_comparison_decadal.ipynb - Model comparison on decadal glacier-wide mass balance and using geodetic observations. <br>
&emsp; ***Figure 10 and C3*** 

3.7_model_comparison_regional.ipynb - Model comparison on time series of glacier-wide mass balance for different regions. <br>
&emsp; ***Figure C2*** 

3.8_plot_mass_balance_gradients.ipynb - Model comparison on mass balance gradients. <br>
&emsp; ***Figure 7*** 

3.9_plot_distributed_mass_balance.ipynb - Plots of predicted distributed mass balance and features for Tunsbergdalsbreen. <br>
&emsp; ***Figure 11*** 

## Note

Filepaths must be adapted to data storage. 

Running 0.4_get_oggm_data.ipynb requires the oggm Python package (oggm_recommended_env.yml) which currently cannot be installed With Windows. 

Running 0.6.1_compute_skyview.py requires the rvt Python package which requires Python 3.7 and must therefore be installed in a separate environment (skyview_env.yml).



