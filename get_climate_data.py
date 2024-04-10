# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:08:04 2023

Get monthly climate data from ERA5-Land for each point (lat/lon).
ERA5-Land monthly data downloaded from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=form

@author: kasj
"""

# Import libraries
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Specify filepaths and filenames

filepath = 'C:/Users/kasj/ML_MB_Norway/Data/'
filename = '2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes.csv'
filename_save = '2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate.csv'

filepath_climate = 'C:/Users/kasj/ML_MB_Norway/Data/ERA5-Land_mon_avg_1950_2022_Norway/'
filename_climate = 'data.nc'

#%% 

# Load data
point_data = pd.read_csv(filepath + filename)#, sep=';')\n",

point_data.drop(columns=point_data.columns[0], axis=1, inplace=True)

#%% Load climate data

with xr.open_dataset(filepath_climate + filename_climate) as ds:
    ds_climate = ds.load()
    
# Get lat and lon from climate dataset

lat = ds_climate.latitude
lon = ds_climate.longitude
    
#%% Load geopotential height
    
with xr.open_dataset(filepath_climate + 'geo.nc') as ds_geo:
    ds_geopotential = ds_geo.load()
    
# Convert geopotential height to geometric height and add to dataset
R_earth = 6367.47 * 10e3 #m (Grib1 radius)
g = 9.81 # m/s2

ds_geopot_metric = ds_geopotential.assign(altitude_climate = lambda ds_geopotential: 
                                          R_earth * ((ds_geopotential.z/g)/(R_earth - (ds_geopotential.z/g))))

# Crop geometric height to grid of climate data
ds_geopot_metric_crop = ds_geopot_metric.sel(longitude = lon, latitude = lat)

#%%

# Reduce expver dimension
ds_climate = ds_climate.reduce(np.nansum, 'expver')

# Dimension expver 1 refers to ERA5 Land, while dimension expver 5 refers to
# ERA5-T (https://confluence.ecmwf.int/display/CUSF/ERA5+CDS+requests+which+return+a+mixture+of+ERA5+and+ERA5T+data)

#ERA5 = xr.open_mfdataset('era5.tp.20200801.nc',combine='by_coords')
#ds_climate_combine =ds_climate.sel(expver=1).combine_first(ds_climate.sel(expver=5))
#%%

#ds_climate_combine.to_netcdf(filepath_climate + 'ERA5_land.nc', format="NETCDF3_CLASSIC")
#ERA5_combine.load()
#ERA5_combine.to_netcdf("era5.tp.20200801.copy.nc")

#%%
# ds_climate contains 14 variables:
# ['u10','v10','t2m','fal','asn','sde','sd','slhf','ssr','str','sshf','ssrd','strd','tp']
# u10: 10m u-component of wind, 
# v10: 10m v-component of wind, 
# d2m: 
# t2m: 2m temperature, 
# fal: Forecast albedo, 
# asn: Snow albedo, 
# sde: Snow depth, 
# sd: Snow depth water equivalent, 
# slhf: Surface latent heat
# ssr: Surface net solar radiation, 
# str: Surface net thermal radiation,
# sshf: Surface sensible heat flux, 
# ssrd: Surface solar radiation downwards, 
# strd: Surface thermal radiation downwards, 
# tp: Total precipitation
# tsn: 
# snowc: Snow cover
# rsn:
# es:
# sf:
# smlt:
# sp

# Go through each location (lat,lon,time) of stake dataset. Find monthly values of the 14 variables
# for each of these locations. This makes 14*12=168 climate variables. 
# We consider the year starts in beginning of october and ends in end of september. Use the 'curr_yr_min_date' to
# get the year and then get monthly values from october (year-1) to september (year).
# For each (lat,lon,time) we need to create an array (1,168) of the 14 monthly variables
# and add these to the dataframe. 

# In addition we could add sum pdds as a variable?

# Get latitude, longitude and year of stakes
#lat_stakes = point_data['lat'].values.round(2)
#lon_stakes = point_data['lon'].values.round(2)
#date_stakes = pd.to_datetime(point_data['dt_curr_year_max_date'].astype('string'), format="%d.%m.%Y %H:%M")
#year_stakes = date_stakes.dt.year.astype('Int64').values

# Create list of variable names from variable names and months.
var_names = list(ds_climate.keys())
month_names = ['_oct','_nov','_dec','_jan','_feb','_mar','_apr','_may','_jun','_jul','_aug','_sep']

# Combine variables and months
month_vars = []
for var in var_names:
    month_vars.extend([f'{var}{mm:02}' for mm in month_names])
    
#%% Get monthly data for each point measurement

# Empty array to store values.
climate_all = np.empty((len(point_data.index),len(month_vars)))
climate_all.fill(np.nan)

altitude_all = np.empty((len(point_data.index),1))
altitude_all.fill(np.nan)

for i in point_data.index:
    
    # Get location and year for point measurement
    lat_stake = point_data.loc[i,'lat'].round(2)#.values.round(2)
    lon_stake = point_data.loc[i,'lon'].round(2)#.values.round(2)
    date_stake = pd.to_datetime(point_data.loc[i,'dt_curr_year_max_date'], format="%d.%m.%Y %H:%M")
    year_stake = date_stake.year
    
    # Select data from climate data
    p_climate = ds_climate.sel(latitude=lat_stake,
                               longitude=lon_stake,
                               time=pd.date_range(str(year_stake-1) + '-09-01',
                                                  str(year_stake) + '-09-01',
                                                  freq='M'),
                               method = "nearest")
    
    # Convert dataarray to dataframe.
    d_climate = p_climate.to_dataframe()
    
    # Drop latitude and longitude columns from dataframe.
    d_climate.drop(columns=['latitude','longitude'],inplace=True)
    
    # Select altitude of climate data for given point
    p_alt = ds_geopot_metric_crop.sel(latitude=lat_stake,
                                      longitude=lon_stake,
                                      method = "nearest")
    
    #d_climate['altitude_climate'] = p_alt.altitude_climate.values[0]
    
    # Flatten dataframe along columns such that each column (oct-sept)
    # follows each other in the flattened array
    a_climate = d_climate.to_numpy().flatten(order='F')
    
    # Store in array.
    climate_all[i,:] = a_climate
    altitude_all[i,:] = p_alt.altitude_climate.values[0]

#%%

# Make pandas dataframe from array with column names from month_vars
df_climate = pd.DataFrame(data = climate_all, columns = month_vars)
df_altitude = pd.DataFrame(data = altitude_all, columns = ['altitude_climate'])

# Concatenate dataframes
df_point_climate = pd.concat([point_data, df_climate, df_altitude], axis=1)#.reindex(point_data.index)

df_point_climate.to_csv(filepath + filename_save, index=False) 

#%%
# Test get data
# Gives two points:
# target_lat = xr.DataArray([60.0,61.0], dims='points')
# target_lon = xr.DataArray([10.0,11.0], dims='points')

# p_climate = ds_climate.sel(latitude=target_lat,
#                            longitude=target_lon,
#                            method = "nearest")
# # Gives four points
# lat = [60.0,61.0]
# lon = [10.0,11.0]
# p_climate = ds_climate.sel(latitude=lat,
#                         longitude=lon,
#                         method='nearest')

# target_year = xr.DataArray([1960,1961])

# P_ERA5 = P_ERA5.sel(time=pd.date_range(str(begin) + '-10-01',
#                                                str(end) + '-09-01',
#                                                freq='M'),method="nearest")

# target_year = 1960
# p_climate = ds_climate.sel(latitude=target_lat,
#                            longitude=target_lon,
#                            time=pd.date_range(str(target_year-1) + '-09-01',
#                                               str(target_year) + '-09-01',
#                                               freq='M'),
#                            method = "nearest")


# year = [1960, 1961]
    
    
    
    
    