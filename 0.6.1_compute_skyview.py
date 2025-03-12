# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import rvt.vis
# import rvt.default
# import numpy as np
# import matplotlib.pyplot as plt


# dem_path = 'C:/Users/kasj/ML_MB_Norway/Data/oggm_data/per_glacier/train/RGI60-08.01126/dem.tif'

# dict_dem = rvt.default.get_raster_arr(dem_path)
# dem_arr = dict_dem["array"]  # numpy array of DEM
# dem_resolution = dict_dem["resolution"]
# dem_res_x = dem_resolution[0]  # resolution in X direction
# dem_res_y = dem_resolution[1]  # resolution in Y direction
# dem_no_data = dict_dem["no_data"]

# # svf, sky-view factor parameters which also applies to asvf and opns
# svf_n_dir = 16  # number of directions
# svf_r_max = 10  # max search radius in pixels
# svf_noise = 0  # level of noise remove (0-don't remove, 1-low, 2-med, 3-high)
# # asvf, anisotropic svf parameters
# asvf_level = 1  # level of anisotropy (1-low, 2-high)
# asvf_dir = 315  # dirction of anisotropy in degrees
# dict_svf = rvt.vis.sky_view_factor(dem=dem_arr, resolution=dem_res_x, compute_svf=True, compute_asvf=True, compute_opns=True,
#                                    svf_n_dir=svf_n_dir, svf_r_max=svf_r_max, svf_noise=svf_noise,
#                                    asvf_level=asvf_level, asvf_dir=asvf_dir,
#                                    no_data=dem_no_data)
# svf_arr = dict_svf["svf"]  # sky-view factor

# svf_path = 'C:/Users/kasj/ML_MB_Norway/Data/oggm_data/per_glacier/train/RGI60-08.01126/svf.tif'
# rvt.default.save_raster(src_raster_path=dem_path, out_raster_path=svf_path, out_raster_arr=svf_arr,
#                         no_data=np.nan, e_type=6)

# import rvt.vis
# import rvt.default
# import numpy as np
# import os

# # Function to process each DEM file and save the SVF output
# def process_dem(dem_path, svf_path):
#     dict_dem = rvt.default.get_raster_arr(dem_path)
#     dem_arr = dict_dem["array"]
#     dem_resolution = dict_dem["resolution"]
#     dem_res_x = dem_resolution[0]
#     dem_res_y = dem_resolution[1]
#     dem_no_data = dict_dem["no_data"]

#     svf_n_dir = 16
#     svf_r_max = 10
#     svf_noise = 0

#     asvf_level = 1
#     asvf_dir = 315

#     dict_svf = rvt.vis.sky_view_factor(dem=dem_arr,
#                                        resolution=dem_res_x,
#                                        compute_svf=True,
#                                        compute_asvf=True,
#                                        compute_opns=True,
#                                        svf_n_dir=svf_n_dir,
#                                        svf_r_max=svf_r_max,
#                                        svf_noise=svf_noise,
#                                        asvf_level=asvf_level,
#                                        asvf_dir=asvf_dir,
#                                        no_data=dem_no_data)

#     svf_arr = dict_svf["svf"]

#     rvt.default.save_raster(src_raster_path=dem_path,
#                             out_raster_path=svf_path,
#                             out_raster_arr=svf_arr,
#                             no_data=np.nan,
#                             e_type=6)

# # Main function to iterate through all subdirectories
# def main(train_dir):
#     for subdir in os.listdir(train_dir):
#         subdir_path = os.path.join(train_dir, subdir)
#         if os.path.isdir(subdir_path) and subdir.startswith('RGI60-08'):
#             dem_path = os.path.join(subdir_path, 'dem.tif')
#             svf_path = os.path.join(subdir_path, 'svf.tif')
#             if os.path.exists(dem_path):
#                 print(f"Processing {dem_path}")
#                 process_dem(dem_path, svf_path)
#             else:
#                 print(f"No DEM found in {subdir_path}")

# if __name__ == '__main__':
#     train_dir = 'C:/Users/kasj/ML_MB_Norway/Data/oggm_data/per_glacier/test'
#     main(train_dir)
    
import rvt.vis
import rvt.default
import numpy as np
import os
import xarray as xr

# Function to process each DEM file and save the SVF output as NetCDF
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

    # Create xarray DataArray
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

    # Save as NetCDF
    svf_da.to_netcdf(svf_path)

# Main function to iterate through all subdirectories
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