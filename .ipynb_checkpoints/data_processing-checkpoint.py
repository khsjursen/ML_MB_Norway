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
