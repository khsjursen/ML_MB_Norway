# ML_MB_Norway
Machine learning for glacier mass balance modelling in Norway.

Repository for manuscript:

Dataset of point mass balance for glaciers in Norway used data used in this study can be found at:
https://doi.org/10.58059/sjse-6w92

In this repository the dataset is found in the Data-folder (please use and refer to the original source above): 
Data/2023-08-28_stake_mb_Hydra2_corrected.csv

0 Processing pipeline:
0.1 data_processing.ipynb - Filter data with missing coordinate/altitude.
    input: 2023-08-28_stake_mb_Hydra2_corrected.csv
    output: 2023-08-28_stake_mb_norway_cleaned.csv
0.2 get_glims_rgi_id.ipynb - Get RGIIDs corresponding to Norwegian glacier ids (breid). Via GLIMS ids.
    input: 2023-08-28_stake_mb_norway_cleaned.csv
    output: 2023-08-28_stake_mb_norway_cleaned_ids.csv
0.3 reproject_coordinates.ipynb - Get lat/lon coordinates corresponding to UTM 32/33/34 coordinates in dataset.
    input: 2023-08-28_stake_mb_norway_cleaned_ids.csv
    output: 2023-08-28_stake_mb_norway_cleaned_ids_latlon.csv
0.4 get_oggm_data.ipynb - For each point/coordinate, get topographical features using the OGGM pipeline using RGIID.
    input: 2023-08-28_stake_mb_norway_cleaned_ids_latlon.csv
    output: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes.csv
0.5 get_climate_data.ipny - For each point/coordinate, get climate variables and altitude of climate data from ERA5-Land.
    Output is complete set of features and targets to be used for training. 
    input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes.csv
    output: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv

1 Data exploration:
1.1 data_exploration.ipynb - Scipt for data exploration.
    input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv

2 Model training and predictions
2.1 final_training.ipynb - Summary notebook with different training cases using annual and seasonal mass balance and different splitting  
    strategies for cross validation.
    input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv
2.2 final_training.ipynb - Summary notebook with different training cases using annual and seasonal mass balance and different splitting  
    strategies for cross validation.
    input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv

3 Analysis and plotting scripts
4.1 
