# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:15:24 2021

@author: kasj

Script to convert from UTM32 to UTM33 coordinates. Used for converting gps
coordinates in NVE point mass balance observations.

"""

#%% Libraries

# Standard libraries
from pyproj import Proj

# External libraries
import numpy as np
import pandas as pd

# Internal libraries

#%% UTM32 to UTM33 conversion

filepath_obs = 'C:/Users/kasj/mass_balance_model/observations/'
filename_mb_obs = 'massbalance_point.csv'

data = pd.read_csv(filepath_obs + filename_mb_obs, sep=';')

x_32 = np.array(data['utm_east'].values)
y_32 = np.array(data['utm_north'].values)

# Define UTM32 projection.
myProj_UTM32 = Proj("+proj=utm +zone=32, \
                    +north +ellps=WGS84 +datum=WGS84 +units=m + no_defs")

# Convert UTM32 coordinates to longitude and latitude.
lon, lat = myProj_UTM32(x_32, y_32, inverse = True)
        
# Define UTM33 projection.
myProj_UTM33 = Proj("+proj=utm +zone=33, \
                    +north +ellps=WGS84 +datum=WGS84 +units=m + no_defs")
        
# Convert lat, lon to UTM33.
x_33, y_33 = myProj_UTM33(lon, lat, inverse=False)

# Insert UTM33 coordinates in DataFrame.
data['utm_east'] = x_33
data['utm_north'] = y_33

data.to_csv(filepath_obs + 'massbalance_point_UTM33.csv', sep=';', index=False)


#%% End of utm_conversion.py


