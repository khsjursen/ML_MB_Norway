# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

from plotting_functions import plot_prediction_per_fold

def select_variables(df_data, vars, *args, drop=False):
    """
    Select or drop variables (features or labels corresponding to dataframe column names) from Pandas DataFrame based 
    on one or more lists of variable names and return DataFrame of selected variables.
    
    Parameters:
    df_data : Pd.DataFrame
        Training or testing dataset.
    vars : list
        List of variables (features or labels corresponding to dataframe column names) to select or drop.
    *args : tuple
        Additional lists of variables.
    drop : bool
        If False (default), select given variables. If True, drop given variables.
    """

    # Make list of all variables
    cols = [y for x in [vars, *args] for y in x]

    # If True, drop variables from dataframe. 
    if drop == True:
        df_data_sel = df_data.drop(cols, axis=1)
    
    #If False, select variables from dataframe.
    else: 
        df_data_sel = df_data[cols]
    
    return df_data_sel

def train_xgb_model(X, y, idc_list, params, scorer='neg_mean_squared_error', return_train=True):

    # Define model object.
    xgb_model = xgb.XGBRegressor()
    
    # Set up grid search. 
    clf = GridSearchCV(xgb_model, 
                       params, 
                       cv=idc_list, # Int or iterator (default for int is kfold)
                       verbose=1, # Controls number of messages
                       n_jobs=4, # No of parallell jobs
                       scoring=scorer, # Can use multiple metrics
                       refit=True, # Default True. For multiple metric evaluation, refit must be str denoting scorer to be used 
                       #to find the best parameters for refitting the estimator.
                       return_train_score=return_train) # Default False. If False, cv_results_ will not include training scores.

    # Fit model to folds
    clf.fit(X, y)

    # Get results of grid search
    print('Cross validation score: ', clf.best_score_)
    print('Grid search best hyperparameters: ', clf.best_params_)

    fitted_model = xgb.XGBRegressor(learning_rate = clf.best_params_['learning_rate'], 
                                    n_estimators = clf.best_params_['n_estimators'],
                                    max_depth = clf.best_params_['max_depth'])
    
    cvl = cross_val_score(fitted_model, X, y, cv=idc_list, scoring='neg_mean_squared_error')

    print('Cross validation scores per fold: ', cvl)
    print('Mean cross validation score: ', cvl.mean())
    print('Standard deviation: ', cvl.std())

    plot_prediction_per_fold(X, y, fitted_model, idc_list)

    return clf, fitted_model, cvl
    