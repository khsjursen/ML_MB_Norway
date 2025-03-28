# Script for training custom loss model on HVL cluster

# Import libraries
import os
import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import dill as pickle

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

from model_functions import reshape_dataset_monthly
#from model_functions import custom_mse_metadata

# Custom objective function scikit learn api with metadata, to be used with custom XGBRegressor class
# Updated based on version from Julian
def custom_mse_metadata(y_true, y_pred, metadata):
    """
    Custom Mean Squared Error (MSE) objective function for evaluating monthly predictions with respect to 
    seasonally or annually aggregated observations.
    
    For use in cases where predictions are done on a monthly time scale and need to be aggregated to be
    compared with the true aggregated seasonal or annual value. Aggregations are performed according to a
    unique ID provided by metadata. The function computes gradients and hessians 
    used in gradient boosting methods, specifically for use with the XGBoost library's custom objective 
    capabilities.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True (seasonally or annually aggregated) values for each instance. For a unique ID, 
        values are repeated n_months times across the group, e.g. the annual mass balance for a group
        of 12 monthly predictions with the same unique ID is repeated 12 times. Before calculating the 
        loss, the mean over the n unique IDs is taken.
    
    y_pred : numpy.ndarray
        Predicted monthly values. These predictions will be aggregated according to the 
        unique ID before calculating the loss, e.g. 12 monthly predictions with the same unique ID is
        aggregated for evaluation against the true annual value.
    
    metadata : numpy.ndarray
        An ND numpy array containing metadata for each monthly prediction. The first column is mandatory 
        and represents the ID of the aggregated group to which each instance belongs. Each group identified 
        by a unique ID will be aggregated together for the loss calculation. The following columns in the 
        metadata can include additional information for each instance that may be useful for tracking or further 
        processing but are not used in the loss calculation, e.g. number of months to be aggregated or the name 
        of the month.
        
        ID (column 0): An integer that uniquely identifies the group which the instance belongs to.
            
    Returns
    -------
    gradients : numpy.ndarray
        The gradient of the loss with respect to the predictions y_pred. This array has the same shape 
        as y_pred.
    
    hessians : numpy.ndarray
        The second derivative (hessian) of the loss with respect to the predictions y_pred. For MSE loss, 
        the hessian is constant and thus this array is filled with ones, having the same shape as y_pred.
    """
            
    # Initialize gradients and hessians
    gradients = np.zeros_like(y_pred)
    hessians = np.ones_like(y_pred)

    # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
    # based on the metadata.
    #y_pred_agg, y_true_mean, grouped_ids, df_metadata = CustomXGBoostRegressor._create_metadata_scores(metadata, y_true, y_pred)
    df_metadata = pd.DataFrame(metadata, columns=['ID', 'N_MONTHS', 'MONTH'])

    # Aggregate y_pred and y_true for each group
    grouped_ids = df_metadata.assign(y_true=y_true, y_pred=y_pred).groupby('ID')
    y_pred_agg = grouped_ids['y_pred'].sum().values
    y_true_mean = grouped_ids['y_true'].mean().values
    
    # Compute gradients
    gradients_agg = y_pred_agg - y_true_mean

    # Create a mapping from ID to gradient
    gradient_map = dict(zip(grouped_ids.groups.keys(), gradients_agg))

    # Assign gradients to corresponding indices
    df_metadata['gradient'] = df_metadata['ID'].map(gradient_map)
    gradients[df_metadata.index] = df_metadata['gradient'].values

    return gradients, hessians

# Define custom XGBRegressor class
# Updated based on version from Julian
class CustomXGBRegressor(XGBRegressor):
    """
    CustomXGBRegressor is an extension of the XGBoost regressor that incorporates additional metadata into the learning process. The estimator
    is tailored to handle training datasets where the last three columns are metadata rather than features.
    
    The metadata is utilized in a custom mean squared error function. This function calculates gradients and hessians incorporating metadata, 
    allowing the model to learn from both standard feature data and additional information provided as metadata.
    
    The custom objective closure captures metadata along with the target values and predicted values to compute the gradients and hessians needed
    for the XGBoost training process.

    The class contains a custom score function (custom mse) that is used in GridSearchCV to evaluate validation performance for each fold.
    This is the default scorer for the class.
    
    Parameters inherited from XGBRegressor are customizable and additional parameters can be passed via kwargs, which will be handled by the
    XGBRegressor's __init__ method.
    
    Examples
    --------
    >>> model = CustomXGBRegressor(n_estimators=500, learning_rate=0.05)
    >>> model.fit(X_train, y_train)  # X_train includes metadata as the last 3 columns
    >>> predictions = model.predict(X_test)  # X_test includes metadata as the last 3 columns
    
    Note: CustomXGBRegressor requires a custom MSE function, `custom_mse_metadata`, which computes the gradient and hessian using additional metadata.
    """
    
    def __init__(self, metadata_shape=3, **kwargs):
        self.metadata_shape = metadata_shape
        super().__init__(**kwargs)

    def fit(self, X, y, **fit_params):
        # Split features from metadata
        metadata, features = X[:, -self.metadata_shape:], X[:, :-self.metadata_shape]

        # Define closure that captures metadata for use in custom objective
        def custom_objective(y_true, y_pred):
            return custom_mse_metadata(y_true, y_pred, metadata)

        # Set custom objective
        self.set_params(objective=custom_objective)

        # Call fit method from parent class (XGBRegressor)
        super().fit(features, y, **fit_params)

        return self

    def predict(self, X):
        # Check if the model is fitted
        check_is_fitted(self)
        
        features = X[:, :-self.metadata_shape]
        
        return super().predict(features)

    def score(self, X, y, sample_weight=None):

        y_pred = self.predict(X)

        metadata, features = X[:, -self.metadata_shape:], X[:, :-self.metadata_shape]
        
        # Get the aggregated predictions and the mean score based on the true labels, and predicted labels
        # based on the metadata.
        df_metadata = pd.DataFrame(metadata, columns=['ID', 'N_MONTHS', 'MONTH'])

        # Aggregate y_pred and y_true for each group
        grouped_ids = df_metadata.assign(y_true=y, y_pred=y_pred).groupby('ID')
        y_pred_agg = grouped_ids['y_pred'].sum().values
        y_true_mean = grouped_ids['y_true'].mean().values

        # Compute mse 
        mse = ((y_pred_agg - y_true_mean) ** 2).mean()

        return -mse # Return negative because GridSearchCV maximizes score

# Get and prepare data 
# Specify filepaths and filenames.
loc = 'local'
type = 'annual' # 'all': all data, 'seasonal': seasonal data, 'annwinter': annual+winter data, 'annsummer': annual+summer, 'annual': only annual, 'summer': only summer, 'winter': only winter

if loc == 'cryocloud':
    filepath = '/home/jovyan/ML_MB_Norway_data/'
elif loc == 'local':
    filepath = 'Data/'
elif loc == 'cluster':
    filepath = '/mirror/khsjursen/ML_MB_Norway/Data/'

filename = '2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate_svf.csv'

# Load data.
data = pd.read_csv(filepath + filename)

# Add year column
data['year']=pd.to_datetime(data['dt_curr_year_max_date'].astype('string'), format="%d.%m.%Y %H:%M")
data['year'] = data.year.dt.year.astype('Int64')

# Remove cells with nan in balance_netto.
glacier_data_annual = data[data['balance_netto'].notna()]
glacier_data_annual.reset_index(drop=True, inplace=True)

glacier_data_winter = data[data['balance_winter'].notna()]
glacier_data_winter.reset_index(drop=True, inplace=True)

glacier_data_summer = data[data['balance_summer'].notna()]
glacier_data_summer.reset_index(drop=True, inplace=True)

test_glaciers = [54, 703, 941, 1135, 1280, 2085, 2320, 2478, 2768, 2769, 3133, 3137, 3138, 3141]

# Get test dataset for each of annual, winter and summer mass balance
df_test_annual = glacier_data_annual[glacier_data_annual['BREID'].isin(test_glaciers)]
df_test_winter = glacier_data_winter[glacier_data_winter['BREID'].isin(test_glaciers)]
df_test_summer = glacier_data_summer[glacier_data_summer['BREID'].isin(test_glaciers)]
# 54 has 189 points
# 703 has 30 points
# 941 has 70 points
# 1280 has 71 points
# 2320 has 83 points
# 2478 has 89 points
# 2769 has 121 points
# 3133 has 38 points
# 3137 has 65 points
# 3138 has 6 points
# 3141 has 72 points

# Get training dataset for each of annual, winter and summer mass balance
df_train_annual = glacier_data_annual[~glacier_data_annual['BREID'].isin(test_glaciers)]
df_train_winter = glacier_data_winter[~glacier_data_winter['BREID'].isin(test_glaciers)]
df_train_summer = glacier_data_summer[~glacier_data_summer['BREID'].isin(test_glaciers)]

# Add number of months to each dataframe
df_train_annual['n_months']=12
df_train_winter['n_months']=7
df_train_summer['n_months']=5
df_test_annual['n_months']=12
df_test_winter['n_months']=7
df_test_summer['n_months']=5

print(f'Train/test annual: {len(df_train_annual)}/{len(df_test_annual)}')
print(f'Train/test winter: {len(df_train_winter)}/{len(df_test_winter)}')
print(f'Train/test summer: {len(df_train_summer)}/{len(df_test_summer)}')
print(f'All train/test: {len(df_train_annual) + len(df_train_winter) + len(df_train_summer)} / {len(df_test_annual) + len(df_test_winter) + len(df_test_summer)}')
print(f'Fraction train/test: {(len(df_train_annual) + len(df_train_winter) + len(df_train_summer)) / (len(df_test_annual) + len(df_test_winter) + len(df_test_summer) + len(df_train_annual) + len(df_train_winter) + len(df_train_summer))} / {(len(df_test_annual) + len(df_test_winter) + len(df_test_summer)) /(len(df_test_annual) + len(df_test_winter) + len(df_test_summer) + len(df_train_annual) + len(df_train_winter) + len(df_train_summer))}')
print(f'Total entries: {len(df_test_annual) + len(df_test_winter) + len(df_test_summer) + len(df_train_annual) + len(df_train_winter) + len(df_train_summer)}')


cols = ['RGIID', 'GLIMSID', 'utm_zone', 'utm_east_approx', 'utm_north_approx', 'altitude_approx', 'location_description', 'location_id', 'stake_no', 'utm_east', 'utm_north', 'dt_prev_year_min_date', 'dt_curr_year_max_date', 'dt_curr_year_min_date', 'stake_remark', 'flag_correction', 'approx_loc', 'approx_altitude', 'diff_north', 'diff_east', 'diff_altitude', 'diff_netto', 'lat_approx', 'lon_approx', 'topo', 'dis_from_border',  'lat', 'lon', 'slope_factor']

snow_depth_m = ['sde_oct','sde_nov','sde_dec','sde_jan','sde_feb','sde_mar','sde_apr','sde_may','sde_jun','sde_jul','sde_aug','sde_sep']
snow_density = ['rsn_oct','rsn_nov','rsn_dec','rsn_jan','rsn_feb','rsn_mar','rsn_apr','rsn_may','rsn_jun','rsn_jul','rsn_aug','rsn_sep']
evaporation = ['es_oct','es_nov','es_dec','es_jan','es_feb','es_mar','es_apr','es_may','es_jun','es_jul','es_aug','es_sep']
snow_cover = ['snowc_oct','snowc_nov','snowc_dec','snowc_jan','snowc_feb','snowc_mar','snowc_apr','snowc_may','snowc_jun','snowc_jul','snowc_aug','snowc_sep']
snow_depth_we = ['sd_oct','sd_nov','sd_dec','sd_jan','sd_feb','sd_mar','sd_apr','sd_may','sd_jun','sd_jul','sd_aug','sd_sep']
snow_temp = ['tsn_oct','tsn_nov','tsn_dec','tsn_jan','tsn_feb','tsn_mar','tsn_apr','tsn_may','tsn_jun','tsn_jul','tsn_aug','tsn_sep']
snow_melt = ['smlt_oct','smlt_nov','smlt_dec','smlt_jan','smlt_feb','smlt_mar','smlt_apr','smlt_may','smlt_jun','smlt_jul','smlt_aug','smlt_sep']
snowfall = ['sf_oct','sf_nov','sf_dec','sf_jan','sf_feb','sf_mar','sf_apr','sf_may','sf_jun','sf_jul','sf_aug','sf_sep']
snow_albedo = ['asn_oct','asn_nov','asn_dec','asn_jan','asn_feb','asn_mar','asn_apr','asn_may','asn_jun','asn_jul','asn_aug','asn_sep']
dewpt_temp = ['d2m_oct','d2m_nov','d2m_dec','d2m_jan','d2m_feb','d2m_mar','d2m_apr','d2m_may','d2m_jun','d2m_jul','d2m_aug','d2m_sep']
surface_pressure = ['sp_oct','sp_nov','sp_dec','sp_jan','sp_feb','sp_mar','sp_apr','sp_may','sp_jun','sp_jul','sp_aug','sp_sep']
sol_rad_net = ['ssr_oct','ssr_nov','ssr_dec','ssr_jan','ssr_feb','ssr_mar','ssr_apr','ssr_may','ssr_jun','ssr_jul','ssr_aug','ssr_sep']
sol_therm_down = ['strd_oct','strd_nov','strd_dec','strd_jan','strd_feb','strd_mar','strd_apr','strd_may','strd_jun','strd_jul','strd_aug','strd_sep']
u_wind = ['u10_oct', 'u10_nov', 'u10_dec', 'u10_jan', 'u10_feb', 'u10_mar', 'u10_apr', 'u10_may', 'u10_jun', 'u10_jul', 'u10_aug', 'u10_sep']
v_wind = ['v10_oct','v10_nov','v10_dec','v10_jan','v10_feb','v10_mar','v10_apr','v10_may','v10_jun','v10_jul','v10_aug','v10_sep']

drop_cols = [y for x in [cols, snow_depth_m, snow_density, evaporation, snow_cover, snow_depth_we, snow_temp, snow_melt, snowfall, snow_albedo, dewpt_temp, surface_pressure, sol_rad_net, sol_therm_down, u_wind, v_wind] for y in x]

# Select features for training
df_train_annual_clean = df_train_annual.drop(drop_cols, axis=1)
df_train_winter_clean = df_train_winter.drop(drop_cols, axis=1)
df_train_summer_clean = df_train_summer.drop(drop_cols, axis=1)
df_train_annual_clean = df_train_annual_clean.drop(['balance_winter','balance_summer'], axis=1)
df_train_winter_clean = df_train_winter_clean.drop(['balance_netto', 'balance_summer'], axis=1)
df_train_summer_clean = df_train_summer_clean.drop(['balance_netto', 'balance_winter'], axis=1)

# Rename target columns to same name
df_train_annual_clean.rename(columns={'balance_netto' : 'balance'}, inplace=True)
df_train_winter_clean.rename(columns={'balance_winter' : 'balance'}, inplace=True)
df_train_summer_clean.rename(columns={'balance_summer' : 'balance'}, inplace=True)

# df_train_X_... now contains columns of all chosen features and column with annual, winter or summer balance

# For summer balance, replace column values in accumulation months with NaN (oct, nov, dec, jan, feb, mar, apr, may)
# For winter balance, replace column values in ablation months with NaN (may, jun, jul, aug, sep, oct)
var = ['t2m', 'sshf', 'slhf', 'ssrd', 'fal','str', 'tp']
mon_summer = ['may', 'jun', 'jul', 'aug', 'sep']
mon_winter = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']

for name in var:
    for mon in mon_summer:
        if mon not in mon_winter:
            df_train_winter_clean[name+'_'+mon]= np.nan
for name in var:   
    for mon in mon_winter:
        if mon not in mon_summer:
            df_train_summer_clean[name+'_'+mon]= np.nan

# Combine all annual, winter and summer data in one dataframe
#df_train_all = pd.concat([df_train_annual_clean, df_train_summer_clean, df_train_winter_clean])

# Use altitude_diff instead of altitude and altitude_climate
df_train_summer_clean['altitude_diff'] = df_train_summer_clean['altitude_climate']-df_train_summer_clean['altitude']
df_train_summer_clean = df_train_summer_clean.drop(['altitude','altitude_climate'], axis=1)
df_train_winter_clean['altitude_diff'] = df_train_winter_clean['altitude_climate']-df_train_winter_clean['altitude']
df_train_winter_clean = df_train_winter_clean.drop(['altitude','altitude_climate'], axis=1)
df_train_annual_clean['altitude_diff'] = df_train_annual_clean['altitude_climate']-df_train_annual_clean['altitude']
df_train_annual_clean = df_train_annual_clean.drop(['altitude','altitude_climate'], axis=1)

df_train_summer_clean = df_train_summer_clean.dropna(axis=1, how='all')
df_train_winter_clean = df_train_winter_clean.dropna(axis=1, how='all')
df_train_annual_clean = df_train_annual_clean.dropna(axis=1, how='all')

n_summer = len(df_train_summer_clean)
n_winter = len(df_train_winter_clean)
n_annual = len(df_train_annual_clean)

df_train_summer_clean.insert(0, 'id', list(range(n_summer)))
df_train_winter_clean.insert(0, 'id', list(range(n_summer, n_summer+n_winter)))
df_train_annual_clean.insert(0, 'id', list(range(n_summer+n_winter, n_summer+n_winter+n_annual)))

# Columns that are not monthly climate variables (identifiers and static variables)
#id_vars = ['id','BREID', 'year', 'altitude','balance','aspect','slope','altitude_climate','n_months']
id_vars = ['id','BREID', 'year', 'altitude_diff','balance','aspect','slope','skyview_factor','n_months']

# Extract the unique variable names and month names from the column names
#variables = set(col.split('_')[0] for col in df.columns if col not in id_vars)
#months = set(col.split('_')[-1] for col in df.columns if col not in id_vars)
variables = ['t2m', 'sshf', 'slhf', 'ssrd', 'fal','str', 'tp']
summer_months_order = ['may', 'jun', 'jul', 'aug', 'sep']
winter_months_order = ['oct','nov','dec','jan','feb','mar','apr']
annual_months_order = ['oct','nov','dec','jan','feb','mar','apr', 'may', 'jun', 'jul', 'aug', 'sep']

# TRAIN MODEL WITH FULL DATASET:

# Reshape dataframes to monthly resolution
df_train_summer_final = reshape_dataset_monthly(df_train_summer_clean, id_vars, variables, summer_months_order)
df_train_winter_final = reshape_dataset_monthly(df_train_winter_clean, id_vars, variables, winter_months_order)
df_train_annual_final = reshape_dataset_monthly(df_train_annual_clean, id_vars, variables, annual_months_order)

# Combine training data in one dataframe
df_train_summer_final.reset_index(drop=True, inplace=True)
df_train_winter_final.reset_index(drop=True, inplace=True)
df_train_annual_final.reset_index(drop=True, inplace=True)

if type == 'annwinter':
    data_list = [df_train_winter_final, df_train_annual_final]
elif type == 'annsummer':
    data_list = [df_train_summer_final, df_train_annual_final]
elif type == 'seasonal':
    data_list = [df_train_winter_final, df_train_summer_final]
elif type == 'annual':
    data_list = [df_train_annual_final]
elif type == 'winter':
    data_list = [df_train_winter_final]
elif type == 'summer':
    data_list = [df_train_summer_final]
else: # all
    data_list = [df_train_summer_final, df_train_winter_final, df_train_annual_final]

df_train_final = pd.concat(data_list)

df_train_final.reset_index(drop=True, inplace=True)

# Select features for training
df_train_X_reduce = df_train_final.drop(['balance','year','BREID'], axis=1)

# Move id and n_months to the end of the dataframe (these are to be used as metadata)
df_train_X = df_train_X_reduce[[c for c in df_train_X_reduce if c not in ['id','n_months','month']] + ['id','n_months','month']]

# Select labels for training
df_train_y = df_train_final[['balance']]

# Get arrays of features+metadata and targets
X_train, y_train = df_train_X.values, df_train_y.values

# USING CUSTOM ITERATOR TO SPLIT ON CYCLE OF YEARS

# Add a 'fold' column to `df_train_final`, initializing to -1
df_train_final['fold'] = -1

# Get sorted list of unique years in the dataset
unique_years = df_train_final['year'].sort_values().unique()

# Round-robin assignment of years to folds
n_folds = 5
for fold, year in enumerate(unique_years):
    df_train_final.loc[df_train_final['year'] == year, 'fold'] = fold % n_folds

# Verify that all rows have been assigned a valid fold
if (df_train_final['fold'] == -1).any():
    raise ValueError("Some rows have not been assigned a valid fold")

# Group by 'id' to maintain groups of rows with the same 'id'
grouped = df_train_final.groupby('id')

# Initialize folds with 5 empty train/test index lists
folds = [([], []) for _ in range(n_folds)]

# Distribute groups into folds
for _, group in grouped:
    fold = group['fold'].iloc[0]  # All rows in group have the same fold
    for fold_idx in range(n_folds):
        if fold == fold_idx:
            folds[fold_idx][1].extend(group.index)  # Assign group to test set of this fold
        else:
            folds[fold_idx][0].extend(group.index)  # Assign group to train set of other folds

# Convert lists to numpy arrays
folds = [(np.array(train_indices), np.array(test_indices)) for train_indices, test_indices in folds]

# Custom cross-validator definition
class CustomFoldIterator(BaseCrossValidator):
    def __init__(self, fold_indices):
        self.fold_indices = fold_indices

    def split(self, X, y=None, groups=None):
        for train_indices, test_indices in self.fold_indices:
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_indices)

# Setup and Execute GridSearchCV (example)
custom_cv = CustomFoldIterator(folds)

# Create splits
splits_s = list(custom_cv.split(X_train, y_train))

# Print number of instances in each split
for i, (train_indices, test_indices) in enumerate(splits_s):
    print(f"Fold {i+1} - Train: {len(train_indices)}, Test: {len(test_indices)}")

# Check fold indices for training/validation data and extract years
fold_indices = []
for i, (train_index, val_index) in enumerate(splits_s):
    print(f"\nFold {i+1}")
    
    # Extract and sort years from train and validation sets
    train_years = np.sort(df_train_final.loc[train_index, 'year'].unique())
    val_years = np.sort(df_train_final.loc[val_index, 'year'].unique())
    
    print(f"TRAIN: {train_index}")
    print(f"VALIDATION: {val_index}")
    print(f"shape(train): {train_index.shape}, shape(validation): {val_index.shape}")
    
    print(f"Number of unique YEARS in TRAIN set: {len(train_years)}, Years: {train_years}")
    print(f"Number of unique YEARS in VALIDATION set: {len(val_years)}, Years: {val_years}")
    
    fold_indices.append((train_index, val_index))

# USING CUSTOM FOLD ITERATOR TO SPLIT ON ID AND YEAR

# Define the year intervals for the folds
#year_intervals = [
#    (1960, 1969),  # Fold 1
#    (1970, 1979),  # Fold 2
#    (1980, 1994),  # Fold 3
#    (1995, 2009),  # Fold 4
#    (2010, 2021)   # Fold 5
#]

#def assign_fold(row):
#    for i, (start_year, end_year) in enumerate(year_intervals):
#        if start_year <= row['year'] <= end_year:
#            return i
#    return -1  # Return -1 if year not in any interval

#df_train_final['fold'] = df_train_final.apply(assign_fold, axis=1)

# Verify that all rows have been assigned a valid fold
#if (df_train_final['fold'] == -1).any():
#    raise ValueError("Some rows have not been assigned a valid fold")

# Group by 'id' to maintain groups of rows with the same 'id'
#grouped = df_train_final.groupby('id')

# Create indices for each fold
#folds = [([], []) for _ in range(5)]  

# Distribute groups into folds
#for _, group in grouped:
#    fold = group['fold'].iloc[0]  # All rows in group have the same fold
#    for fold_idx in range(5):
#        if fold == fold_idx:
#            folds[fold_idx][1].extend(group.index)  # Assign group to test set of this fold
#        else:
#            folds[fold_idx][0].extend(group.index)  # Assign group to train set of other folds

# Convert lists to numpy arrays
#folds = [(np.array(train_indices), np.array(test_indices)) for train_indices, test_indices in folds]

# Define the custom cross-validator
#class CustomFoldIterator(BaseCrossValidator):
#    def __init__(self, fold_indices):
#        self.fold_indices = fold_indices
#
#    def split(self, X, y=None, groups=None):
#        for train_indices, test_indices in self.fold_indices:
#            yield train_indices, test_indices

#    def get_n_splits(self, X=None, y=None, groups=None):
#        return len(self.fold_indices)

#custom_cv = CustomFoldIterator(folds)

# Create splits
#splits_s = list(custom_cv.split(X_train, y_train))

# Convert Int64Index to numpy arrays 
#splits_s = [(np.array(train_indices), np.array(test_indices)) for (train_indices, test_indices) in splits_s]

#for i, (train_indices, test_indices) in enumerate(splits_s):
#    print(f"Fold {i+1} - Train: {len(train_indices)}, Test: {len(test_indices)}")

# Check fold indices for training/validation data
#fold_indices = []

#for i, (train_index, val_index) in enumerate(splits_s):
#    print(f"Fold {i+1}")
#    print("TRAIN:", train_index)
#    print("VALIDATION:", val_index)
#    print("shape(train):", train_index.shape, "shape(validation):", val_index.shape)
#    fold_indices.append((train_index, val_index))

# USE GROUPKFOLD TO SPLIT ON ID

# Get glacier IDs from training dataset (in the order of which they appear in training dataset).
# gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
#gp_s = np.array(df_train_final['id'].values)

# Use five folds
#group_kf = GroupKFold(n_splits=5)

# Split into folds according to group by glacier ID.
# For each unique glacier ID, indices in gp_s indicate which rows in X_train_s and y_train_s belong to the glacier.
#splits_s = list(group_kf.split(X_train, y_train, gp_s))

#print(len(gp_s))
print(y_train.shape)
print(X_train.shape)
print(df_train_X.columns)
print(df_train_y.columns)

# HYPERPARAMETER TUNING

# Define hyperparameter grid
param_ranges = {'max_depth': [3, 4, 5, 6, 7], # Depth of tree
                'n_estimators': [100, 200, 300, 400, 500], # Number of trees (too many = overfitting, too few = underfitting)
                'learning_rate': [0.01, 0.05, 0.1, 0.15], #[0,1]
                'gamma': [0], # Regularization parameter, minimum loss reduction required to make split [0,inf]
                #'lambda': [0, 10], # Regularization [1,inf]
                #'alpha': [0, 10], # Regularization [0,inf]
                #'colsample_bytree': [0.5, 1], # (0,1]  A smaller colsample_bytree value results in smaller and less complex models, which can help prevent overfitting. It is common to set this value between 0.5 and 1.
                #'subsample': [0.5, 1], # (0,1] common to set this value between 0.5 and 1
                'min_child_weight': [0, 5, 10], # [0,inf]
                'random_state': [23]
               } 

xgb_model = CustomXGBRegressor()

n_jobs = 20

clf = GridSearchCV(xgb_model, 
                   param_ranges, 
                   cv=splits_s,
                   verbose=2, 
                   n_jobs=n_jobs, 
                   scoring = None, # Uses default in CustomXGBRegressor()
                   refit=True, 
                   error_score='raise',
                   return_train_score=True) # Default False. If False, cv_results_ will not include training scores.

clf.fit(X_train, y_train)

# Save the model to a binary file
best_model = clf.best_estimator_

# Print cv search time
mean_fit_time= clf.cv_results_['mean_fit_time']
mean_score_time= clf.cv_results_['mean_score_time']
n_splits  = clf.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(clf.cv_results_).shape[0] #Iterations per split

print('Total search time (seconds): ', (np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter)/n_jobs)

# Create folder to store results
filepath_save = 'Training_cluster/'
dir = os.path.join(filepath_save, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(dir)

# Save cv-objects
best_model.save_model(dir + '/' + 'custom_loss_best_model.json')
# Save the model using dill
with open(dir + '/' + 'custom_loss_cv_grid.pkl', 'wb') as f:
    pickle.dump(clf, f)
#joblib.dump(clf, dir + '/' + 'custom_loss_cv_grid.pkl')
#joblib.dump(best_model, dir + '/' + 'custom_loss_best_model.pkl')


