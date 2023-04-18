# -*- coding: utf-8 -*-
"""
KHS 03.03.2023

Preprocessing point MB data.

SCRIPT DESCRIPTION

"""

#%% Libraries

# Standard libraries

# External libraries
import numpy as np
import xarray as xr
import pandas as pd

# Internal libraries

#%% Read files

# Specify filepaths and filenames.
filepath = 'C:/Users/kasj/OneDrive - Høgskulen på Vestlandet/Data/NVE_stake_data/'
filename = '2022-10-12_stake_mb_Hydra2_corrected.csv'

# Load data.
data = pd.read_csv(filepath + filename, sep=';')

# Rename columns.
data = data.rename(columns={"utm_east3": "utm_east_approx", 
                            "utm_north4": "utm_north_approx", 
                            "altitude5": "altitude_approx"})

# New columns indicating if location/altitude is approximate. Fill new column with "N" for
# location/altitude is not approximate.
data['approx_loc'] = 'N'
data['approx_altitude'] = 'N'

# If "utm_east" values are missing, fill column "approx_loc" with "Y" indicating
# that the location is approximate.
data.loc[data['utm_east'].isna(), 'approx_loc'] = 'Y'

# If "altitude" values are missing, fill column "approx_altitude" with "Y"
# indicating that the altitude is approximate.
data.loc[data['altitude'].isna(), 'approx_altitude'] = 'Y'

# Where there is no exact location, fill inn approximate location based on
# "utm_east_approx" and "utm_north_approx" in columns "utm_east" and "utm_north".
# Location is now filled for every observation, with column "approx_loc" indicating
# wether location is approximate (Y) or exact (N).
approx_locs_east = data.loc[data['approx_loc'] == 'Y', 'utm_east_approx']
data.loc[data['approx_loc'] == 'Y', 'utm_east'] = approx_locs_east
approx_locs_north = data.loc[data['approx_loc'] == 'Y', 'utm_north_approx']
data.loc[data['approx_loc'] == 'Y', 'utm_north'] = approx_locs_north

# Same operation with missing altitude, fill in values from "altitude_approx"
# in column "altitude". 
approx_alt = data.loc[data['approx_altitude'] == 'Y', 'altitude_approx']
data.loc[data['approx_altitude'] == 'Y', 'altitude'] = approx_alt 

# Calculate difference between approximate and exact positions and altitude as
# a measure of precision/quality of approximate locations.
data['diff_north'] = data['utm_north'] - data['utm_north_approx']
data['diff_east'] = data['utm_east'] - data['utm_east_approx']
data['diff_altitude'] = data['altitude'] - data['altitude_approx']

# 4194 of 4201 points. A total of 7 rows are missing both altitude and altitude_approx.
data_crop_alt = data[data['diff_altitude'].notna()]

# 4053 of 4201 points. A total of 148 rows are missing both exact loc and approx loc.
data_crop_loc = data[data['diff_east'].notna()]

# Cleaned dataset with 4046 instances. A total of 155 points are either missing 
# both exact and approximate coordinates or altitude.
data_crop = data_crop_alt[data_crop_alt['diff_east'].notna()]

# Check balances:
data_crop['diff_netto'] = data_crop['balance_netto'] - (data_crop['balance_winter'] + data_crop['balance_summer'])

# Save cropped dataset
#data_crop.to_csv("C:/Users/kasj/ML_MB_Norway/Data/stake_mb_norway_cleaned.csv")

#%% Stats

# Stats for quality of approximate locations and altitudes:
mean_loc_diff_east = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_east'].mean()
min_loc_diff_east = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_east'].min()
max_loc_diff_east = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_east'].max()
sd_loc_diff_east = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_east'].std()

mean_loc_diff_north = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_north'].mean()
min_loc_diff_north = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_north'].min()
max_loc_diff_north = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_north'].max()
sd_loc_diff_north = data_crop.loc[data_crop['approx_loc'] != 'Y', 'diff_north'].std()

mean_altitude_diff = data_crop.loc[data_crop['approx_altitude'] != 'Y', 'diff_altitude'].mean()
min_altitude_diff = data_crop.loc[data_crop['approx_altitude'] != 'Y', 'diff_altitude'].min()
max_altitude_diff = data_crop.loc[data_crop['approx_altitude'] != 'Y', 'diff_altitude'].max()
sd_altitude_diff = data_crop.loc[data_crop['approx_altitude'] != 'Y', 'diff_altitude'].std()

# Number of instances of winter balance: 3680
data_crop['balance_winter'].notna().sum()

# Number of instances of summer balance: 3805
data_crop['balance_summer'].notna().sum()

# Number of instances of annual balance: 3839
data_crop['balance_netto'].notna().sum()

# Number of points with exact coordinates: 3717 of 4046
no_exact_locs = (data_crop['approx_loc'].values == 'N').sum()

# Number of points with approximate coordinates: 329 of 4046
no_approx_locs = (data_crop['approx_loc'].values == 'Y').sum()
    
# Number of points with exact altitude: 4009
no_exact_alt = (data_crop['approx_altitude'].values == 'N').sum()

# Number of points with exact altitude: 37
no_approx_alt = (data_crop['approx_altitude'].values == 'Y').sum()

# Number of points w/o exact coordinates or altitude: 32
no_approx_both = ((data_crop['approx_altitude'].values == 'Y') & (data_crop['approx_loc'].values == 'Y')).sum()
   
# Number of unique glacier IDs: 32
no_unique_id = data_crop['glacier_id'].nunique()    

# List of unique glacier IDs:
list_unique_id = list(data_crop['glacier_id'].unique())

# Unique glacier IDs with number of entries per ID.
# Index is glacier ID and column is number of entries per glacier ID.
len_rec_per_id = data_crop['glacier_id'].value_counts().to_frame()    




