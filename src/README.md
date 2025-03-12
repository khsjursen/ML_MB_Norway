# Notebooks

The point mass balance dataset is found in the Data-folder: 
Data/2023-08-28_stake_mb_Hydra2_corrected.csv

Please use and refer to the original source:
https://doi.org/10.58059/sjse-6w92

### 0 Processing pipeline: \
0.1 data_processing.ipynb - Filter data with missing coordinate/altitude. \
0.2 get_glims_rgi_id.ipynb - Get RGIIDs corresponding to Norwegian glacier ids (breid). Via GLIMS ids. \
0.3 reproject_coordinates.ipynb - Get lat/lon coordinates corresponding to UTM 32/33/34 coordinates in dataset. \
0.4 get_oggm_data.ipynb - For each point/coordinate, get topographical features using the OGGM pipeline using RGIID. \
0.5 get_climate_data.ipny - For each point/coordinate, get climate variables and altitude of climate data from ERA5-Land. \
0.6.1 and 0.6.2 add skyview factor - Compute and add the skyview factor as a feature. \

### 1 Data exploration: \
1.1 data_exploration.ipynb - Scipt for data exploration. \

### 2 Model training and predictions \
2.1.1 model_training_validation.ipynb - Set up test and training datasets and cross validation. \ 
2.1.2 model_training_cluster.ipynb - Hyperparameter tuning using cross validation scheme. \ 
2.2 model_predictions.ipynb - Predict mass Balance in each grid cell for each glacier and month using trained model. \

### 3 Analysis and plotting scripts \
