# ML_MB_Norway
Machine learning of mass balance of glaciers in Norway.

Raw, quality checked dataset of point mass balance for 32 glaciers in Norway from Hydra2 database: Data/2023-08-28_stake_mb_Hydra2_corrected.csv

Processing pipeline:

1. data_processing.ipynb - Filter data with missing coordinate/altitude.
   input: 2023-08-28_stake_mb_Hydra2_corrected.csv
   output: 2023-08-28_stake_mb_norway_cleaned.csv
2. get_glims_rgi_id.ipynb - Get RGIIDs corresponding to Norwegian glacier ids (breid). Via GLIMS ids.
   input: 2023-08-28_stake_mb_norway_cleaned.csv
   output: 2023-08-28_stake_mb_norway_cleaned_ids.csv
3. reproject_coordinates.ipynb - Get lat/lon coordinates corresponding to UTM 32/33/34 coordinates in dataset.
   input: 2023-08-28_stake_mb_norway_cleaned_ids.csv
   output: 2023-08-28_stake_mb_norway_cleaned_ids_latlon.csv
4. get_oggm_data.ipynb - For each point/coordinate, get topographical features using the OGGM pipeline using RGIID.
   input: 2023-08-28_stake_mb_norway_cleaned_ids_latlon.csv
   output: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes.csv
5. get_climate_data.py - For each point/coordinate, get climate variables and altitude of climate data from ERA5-Land.
   Output is complete set of features and targets to be used for training. 
   input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes.csv
   output: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv

Data exploration:
- data_exploration.ipynb - Scipt for data exploration.
  input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv

Model training
- final_training.ipynb - Summary notebook with different training cases using annual and seasonal mass balance and different splitting strategies for cross validation.
  input: 2023-08-28_stake_mb_norway_cleaned_ids_wattributes_climate.csv
