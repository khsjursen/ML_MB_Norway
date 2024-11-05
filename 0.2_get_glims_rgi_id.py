# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:41:04 2021

@author: kasj

Script to get GLIMS and RGI ID based on NVE BREID.

"""
#%% Libraries

# Standard libraries

# External libraries
import geopandas as gpd
import pandas as pd

# Internal libraries

#%% Get RGI and GLIMS IDs based on list of NVE BREID

# Filepaths and filenames.
file_mb_data = 'C:/Users/kasj/ML_MB_Norway/Data/stake_mb_norway_cleaned.csv'
filepath_shapefiles = 'C:/Users/kasj/ML_MB_Norway/Data/Shape_files/'
#gl_id_file = 'glacier_id_JOB_only_BREID.txt'

# Read list of glacier IDs (BREID) as dataframe and get list of BREID.
mb_data = pd.read_csv(file_mb_data, sep=',')
breid_list = mb_data['glacier_id'].unique().tolist()

#%%
def getrgiid(id_list, save=False, **kwargs):

    # File directories and file names.
    #filedir_breid_shp = '/mirror/khsjursen/mass_balance_model/shape_files/'
    shp_file_breid = 'cryoclim_GAO_NO_1999_2006_UTM_33N.shp'

    #filedir_rgi_shp = '/mirror/khsjursen/mass_balance_model/shape_files/RGI_Data/'
    shp_file_rgi = '08_rgi60_Scandinavia.shp'

    # Read shape file containing BREID and corresponding GLIMSID as dataframe.
    # Crop dataframe with values of BREID in list of BREID. Drop all columns 
    # except BREID, BRENAVN (glacier name), HOVEDBREAK (main glacier complex
    # acronym) and GLIMSID. 
    df_breid = gpd.read_file(filepath_shapefiles + shp_file_breid)
    df_breid_cropped = df_breid[df_breid['BREID'].isin(id_list)]
    df_breid_glimsid = df_breid_cropped[['BREID','BRENAVN','HOVEDBREAK','GLIMSID']]

    # Sort the dataframe with BREID and corresponding GLIMSID in the order
    # of id_list. 
    df_out = df_breid_glimsid.set_index('BREID').loc[id_list].reset_index(inplace=False)

    # Get list of GLIMSID. 
    glims_list = df_out['GLIMSID'].values.tolist()

    # Read shape file containing GLIMSID and corresponding RGIID as dataframe.
    df_rgi = gpd.read_file(filepath_shapefiles + shp_file_rgi)

    # Get list of RGI ID corresponding to list of GLIMS ID from df_rgi. Dataframe
    # df_rgi contains more values than df_breid, so selection by boolean mask
    # cannot be used. Instead, df_rgi is cropped to contain only rows with GLIMSId
    # from glims_list. Then values are sorted so that the GLIMSId column is in the
    # same order as glims_list. RGIIds are then selected (and are now in the
    # correct order with respect to id_list and glims_list).
    df_rgi_cropped = df_rgi[df_rgi['GLIMSId'].isin(glims_list)]
    df_rgi_cropped.GLIMSId = df_rgi_cropped.GLIMSId.astype("category")
    df_rgi_cropped.GLIMSId.cat.set_categories(glims_list, inplace = True)
    df_rgi_cropped = df_rgi_cropped.sort_values(["GLIMSId"])
    rgi_list = df_rgi_cropped['RGIId'].values.tolist()

    # Add list of RGI ID to dataframe
    # Dataframe now contains breid, Glims ID and RGI ID
    df_out['RGIID'] = rgi_list
    
    # Dictionaries of Glims ID and RGI id with breid as keys.
    breid_rgi = pd.Series(df_out.RGIID.values,index=df_out.BREID).to_dict()
    breid_glims = pd.Series(df_out.GLIMSID.values, index=df_out.BREID).to_dict()
    
    # Map Glims ID and RGI ID to glacier_id in mb_data.
    mb_data['RGIID'] = mb_data["glacier_id"].map(breid_rgi)
    mb_data['GLIMSID'] = mb_data["glacier_id"].map(breid_glims)

    # Rename Norwegian Id column to BREID
    mb_data.rename(columns={'glacier_id':'BREID'}, inplace=True)
    
    # Move ids to front of dataframe
    glims = mb_data.pop('GLIMSID')
    mb_data.insert(0, 'GLIMSID', glims)
    rgi = mb_data.pop('RGIID')
    mb_data.insert(0, 'RGIID', rgi)   

    # Drop old index column
    mb_data.drop(mb_data.columns[2], axis=1, inplace=True)

    if save==True:

        # Get name of new file to store ids.
        new_gl_id_file = kwargs.get('filename_new', None)

        # Save new dataframe with all IDs.
        mb_data.to_csv('C:/Users/kasj/ML_MB_Norway/Data/' + new_gl_id_file, sep=';', index=None)

    # Return dataframe with all ids included.
    return(mb_data)

mb_data_processed = getrgiid(breid_list, save=True, filename_new='new_gl_id.txt')


#%% End of get_glims_rgi_id.py