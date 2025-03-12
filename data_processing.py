# Author: Kamilla Hauknes Sjursen (kasj@hvl.no) November 2024

import numpy as np
import pandas as pd
import warnings

def get_ann_seas_train_test_datasets(data, test_glaciers):
    
    # Add year column
    data['year']=pd.to_datetime(data['dt_curr_year_max_date'].astype('string'), format="%d.%m.%Y %H:%M")
    data['year'] = data.year.dt.year.astype('Int64')

    # Remove cells with nan in balance_netto
    glacier_data_annual = data[data['balance_netto'].notna()]
    glacier_data_annual.reset_index(drop=True, inplace=True)

    glacier_data_winter = data[data['balance_winter'].notna()]
    glacier_data_winter.reset_index(drop=True, inplace=True)

    glacier_data_summer = data[data['balance_summer'].notna()]
    glacier_data_summer.reset_index(drop=True, inplace=True)

    # Get test dataset for each of annual, winter and summer mass balance
    df_test_annual = glacier_data_annual[glacier_data_annual['BREID'].isin(test_glaciers)]
    df_test_winter = glacier_data_winter[glacier_data_winter['BREID'].isin(test_glaciers)]
    df_test_summer = glacier_data_summer[glacier_data_summer['BREID'].isin(test_glaciers)]

    # Get training dataset for each of annual, winter and summer mass balance
    df_train_annual = glacier_data_annual[~glacier_data_annual['BREID'].isin(test_glaciers)]
    df_train_winter = glacier_data_winter[~glacier_data_winter['BREID'].isin(test_glaciers)]
    df_train_summer = glacier_data_summer[~glacier_data_summer['BREID'].isin(test_glaciers)]

    # Add number of months to each dataframe
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        df_train_annual['n_months']=12
        df_train_winter['n_months']=7
        df_train_summer['n_months']=5
        df_test_annual['n_months']=12
        df_test_winter['n_months']=7
        df_test_summer['n_months']=5

    print(f'Train/test annual: {len(df_train_annual)}/{len(df_test_annual)}')
    print(f'Train/test winter: {len(df_train_winter)}/{len(df_test_winter)}')
    print(f'Train/test summer: {len(df_train_summer)}/{len(df_test_summer)}')
    print(f'All train/test: {len(df_train_annual)+len(df_train_winter)+len(df_train_summer)} / {len(df_test_annual)+len(df_test_winter)+len(df_test_summer)}')
    print(f'Fraction train/test: {(len(df_train_annual)+len(df_train_winter)+len(df_train_summer)) / (len(df_test_annual)+len(df_test_winter)+len(df_test_summer)+len(df_train_annual)+len(df_train_winter)+len(df_train_summer))} / {(len(df_test_annual)+len(df_test_winter)+len(df_test_summer)) /(len(df_test_annual)+len(df_test_winter)+len(df_test_summer) + len(df_train_annual)+len(df_train_winter)+len(df_train_summer))}')
    print(f'Total entries: {len(df_test_annual)+len(df_test_winter)+len(df_test_summer) + len(df_train_annual)+len(df_train_winter)+len(df_train_summer)}')

    datasets_train = {'train_annual': df_train_annual,
                      'train_winter': df_train_winter,
                      'train_summer': df_train_summer}
    datasets_test = {'test_annual': df_test_annual,
                     'test_winter': df_test_winter,
                     'test_summer': df_test_summer}
    
    return datasets_train, datasets_test

    
# Prepare features for dataset
def prepare_features(df, season, id_vars_ext, climate_vars, drop_altitude=True):
    
    # Select features for training
    cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in climate_vars) or col in id_vars_ext]
    df_clean = df[cols]

    # For summer balance, replace column values in accumulation months with NaN (oct, nov, dec, jan, feb, mar, apr, may)
    # For winter balance, replace column values in ablation months with NaN (may, jun, jul, aug, sep, oct)
    mon_summer = ['may', 'jun', 'jul', 'aug', 'sep']
    mon_winter = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']

    # Get target column and rename to same name
    if season == 'annual':
        df_clean = df_clean.drop(['balance_winter','balance_summer'], axis=1)
        df_clean.rename(columns={'balance_netto' : 'balance'}, inplace=True)
        
    elif season == 'winter':
        df_clean = df_clean.drop(['balance_netto', 'balance_summer'], axis=1)
        df_clean.rename(columns={'balance_winter' : 'balance'}, inplace=True)

        for name in climate_vars:
            for mon in mon_summer:
                if mon not in mon_winter:
                    df_clean[name+mon]= np.nan
        
    elif season == 'summer':
        df_clean = df_clean.drop(['balance_netto', 'balance_winter'], axis=1)
        df_clean.rename(columns={'balance_summer' : 'balance'}, inplace=True)

        for name in climate_vars:   
            for mon in mon_winter:
                if mon not in mon_summer:
                    df_clean[name+mon]= np.nan

    # Use altitude_diff instead of altitude and altitude_climate
    df_clean['altitude_diff'] = df_clean['altitude_climate']-df_clean['altitude']

    if drop_altitude==True:
        df_clean = df_clean.drop(['altitude','altitude_climate'], axis=1)

    df_clean = df_clean.dropna(axis=1, how='all')

    return df_clean


# Normalize features
# Using min-max scaling
#from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
#scaler = MinMaxScaler()

# Extract metadata columns
#metadata_columns = X_train_unnorm[:, -3:]

# Extract remaining columns
#remaining_columns = X_train_unnorm[:, :-3]

# Apply MinMaxScaler to the remaining columns
#scaled_remaining_columns = scaler.fit_transform(remaining_columns)

# Combine scaled columns with metadata columns
#X_train = np.hstack((scaled_remaining_columns, metadata_columns))

# Apply to validation/test data
#X_val_scaled = scaler.transform(X_val)
#X_test_scaled = scaler.transform(X_test)


# Normalize features
# Using min-max scaling of training data

# Extract metadata columns
#metadata_columns_test = X_test_unnorm[:, -3:]

# Extract remaining columns
#remaining_columns_test = X_test_unnorm[:, :-3]

# Apply MinMaxScaler to the remaining columns
#scaled_remaining_columns_test = scaler.transform(remaining_columns_test)

# Combine scaled columns with metadata columns
#X_test = np.hstack((scaled_remaining_columns_test, metadata_columns_test))