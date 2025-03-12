# -*- coding: utf-8 -*-
"""
Script to compute skyview factor on DEM.
The rvt package requires Python 3.7 and can be installed and run using the skyview_env.yml environment. 
"""

# Import libraries
import rvt.vis
import rvt.default
import numpy as np
import os
import xarray as xr

# Function to process each DEM and save the SVF output as netcdf
def process_dem(dem_path, svf_path):
    dict_dem = rvt.default.get_raster_arr(dem_path)
    dem_arr = dict_dem["array"]
    dem_resolution = dict_dem["resolution"]
    dem_res_x = dem_resolution[0]
    dem_res_y = dem_resolution[1]
    dem_no_data = dict_dem["no_data"]

    svf_n_dir = 16
    svf_r_max = 10
    svf_noise = 0

    asvf_level = 1
    asvf_dir = 315

    dict_svf = rvt.vis.sky_view_factor(dem=dem_arr,
                                       resolution=dem_res_x,
                                       compute_svf=True,
                                       compute_asvf=True,
                                       compute_opns=True,
                                       svf_n_dir=svf_n_dir,
                                       svf_r_max=svf_r_max,
                                       svf_noise=svf_noise,
                                       asvf_level=asvf_level,
                                       asvf_dir=asvf_dir,
                                       no_data=dem_no_data)

    svf_arr = dict_svf["svf"]

    svf_da = xr.DataArray(
        svf_arr,
        dims=["y", "x"],
        coords={
            "y": np.arange(dem_arr.shape[0]) * dem_res_y,
            "x": np.arange(dem_arr.shape[1]) * dem_res_x
        },
        attrs=dict(
            description="Sky-View Factor",
            units="unitless",
        )
    )

    svf_da.to_netcdf(svf_path)

# Iterate through glacier directories
def main(train_dir):
    for subdir in os.listdir(train_dir):
        subdir_path = os.path.join(train_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith('RGI60-08'):
            dem_path = os.path.join(subdir_path, 'dem.tif')
            svf_path = os.path.join(subdir_path, 'svf.nc')
            if os.path.exists(dem_path):
                print(f"Processing {dem_path}")
                process_dem(dem_path, svf_path)
            else:
                print(f"No DEM found in {subdir_path}")

if __name__ == '__main__':
    train_dir = 'C:/Users/kasj/ML_MB_Norway/Data/oggm_data/per_glacier/test'
    main(train_dir)