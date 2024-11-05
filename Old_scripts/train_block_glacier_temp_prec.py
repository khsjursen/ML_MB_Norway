# Script for training blocking-by-glacier models on HVL cluster

# Import libraries
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import joblib

from model_functions import select_variables
from model_functions import train_xgb_model_no_plot

# Specify filepaths and filenames.
loc = 'cluster'

if loc == 'cryocloud':
    filepath = '/home/jovyan/ML_MB_Norway_data/'
elif loc == 'local':
    filepath = 'Data/'
elif loc == 'cluster':
    filepath = '/mirror/khsjursen/ML_MB_Norway/Data/'

filename = '2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate.csv'

# Load data.
data = pd.read_csv(filepath + filename)

# Add year column
data['year']=pd.to_datetime(data['dt_curr_year_max_date'].astype('string'), format="%d.%m.%Y %H:%M")
data['year'] = data.year.dt.year.astype('Int64')

# Remove cells with nan in balance_netto.
glacier_data_all = data[data['balance_netto'].notna()]
glacier_data_all.reset_index(drop=True, inplace=True)

# Manually select test glaciers
test_glaciers = [54, 703, 941, 1135, 1280, 2085, 2320, 2768, 2478, 2769, 3127, 3141]

df_test = glacier_data_all[glacier_data_all['BREID'].isin(test_glaciers)]
# 54 has 189 points
# 703 has 30 points
# 941 has 70 points
# 1280 has 71 points
# 2085 has 163 points
# 2320 has 83 points
# 2768 has 12 points
# 2478 has 89 points
# 2769 has 121 points
# 3127 has 145 points
# 3141 has 72 points
# Test dataset has 1101 points in total, approximately 28% of the data
# Train dataset has 2809 points, approximately 72% of the data

df_train = glacier_data_all[~glacier_data_all['BREID'].isin(test_glaciers)]

# Select features
base_cols = ['altitude','aspect','slope','altitude_climate','lat','lon']
temp_cols = ['t2m_oct','t2m_nov','t2m_dec','t2m_jan','t2m_feb','t2m_mar','t2m_apr','t2m_may','t2m_jun','t2m_jul','t2m_aug','t2m_sep']
prec_cols = ['tp_oct','tp_nov','tp_dec','tp_jan','tp_feb','tp_mar','tp_apr','tp_may','tp_jun','tp_jul','tp_aug','tp_sep']
label_cols = ['balance_netto']

# Shuffle training dataset in order to shuffle glaciers such that they do not appear in order of glacier ID
df_train_s = df_train.sample(frac=1, random_state=5)
df_train_s.reset_index(drop=True, inplace=True)

# Select features for training
df_train_X_s = select_variables(df_train_s, base_cols, temp_cols, prec_cols)

# Select labels for training
df_train_y_s = select_variables(df_train_s, label_cols)
X_train_s, y_train_s = df_train_X_s.values, df_train_y_s.values

# Get glacier IDs from training dataset (in the order of which they appear in training dataset).
# gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
gp_s = np.array(df_train_s['BREID'].values)
print(gp_s)
print(gp_s.shape)
print(X_train_s.shape)

# Use five folds
group_kf = GroupKFold(n_splits=5)

# Split into folds according to group by glacier ID.
# For each unique glacier ID, indices in gp_s indicate which rows in X_train_s and y_train_s belong to the glacier.
splits_s = list(group_kf.split(X_train_s, y_train_s, gp_s))

# Define parameter ranges.
param_ranges = {'max_depth': [2, 3, 4, 5, 6, 7, 8], # Depth of tree
                'n_estimators': [50, 100, 200, 300, 400, 500], # Number of trees (too many = overfitting, too few = underfitting)
                'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], #[0,1]
                'gamma': [0, 1, 5, 10], # Regularization parameter [0,inf]
                'lambda': [0, 1, 5, 10], # Regularization [1,inf]
                'alpha': [0, 1, 5, 10], # Regularization [0,inf]
                'colsample_bytree': [0.5, 0.75, 1], # (0,1]  A smaller colsample_bytree value results in smaller and less complex models, which can help prevent overfitting. It is common to set this value between 0.5 and 1.
                'subsample': [0.5, 0.75, 1], # (0,1] common to set this value between 0.5 and 1
                'min_child_weight': [0, 1, 5, 10], # [0,inf]
                'random_state': [23]
               } 
#param_ranges = {'max_depth' : [2, 3, 4, 5, 6, 7, 8],
#                'n_estimators' : [50, 100, 200, 300],
#                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#}

n_jobs = 40

# Train model
cv_grid, best_model, cvl_scores = train_xgb_model_no_plot(X_train_s, y_train_s, splits_s, param_ranges, n_jobs=n_jobs, scorer='neg_mean_squared_error')

# Print cv search time
mean_fit_time= cv_grid.cv_results_['mean_fit_time']
mean_score_time= cv_grid.cv_results_['mean_score_time']
n_splits  = cv_grid.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(cv_grid.cv_results_).shape[0] #Iterations per split

print('Total search time (seconds): ', (np.mean(mean_fit_time + mean_score_time) * n_splits * n_iter)/n_jobs)

# Create folder to store results
filepath_save = '/mirror/khsjursen/ML_MB_Norway/Models/Block_glacier_5fold/'
dir = os.path.join(filepath_save, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(dir)

# Save cv-objects
joblib.dump(cv_grid, dir + '/' + 'temp_prec_cv_grid.pkl')
joblib.dump(cvl_scores, dir + '/' + 'temp_prec_cv_scores.pkl')
joblib.dump(best_model, dir + '/' + 'temp_prec_cv_best_model.pkl')