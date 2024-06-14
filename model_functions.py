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

def get_model_residuals(X, y, model, idc_list):

    y_test_list = []
    y_pred_list = []

    for train_index, test_index in idc_list:
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_list.extend(y_test)
        y_pred_list.extend(y_pred)
        
    # Arrays of predictions and observations for each fold
    y_test_all = np.hstack([*y_test_list])
    y_pred_all = np.hstack([*y_pred_list])

    return(y_test_all - y_pred_all)

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
                       verbose=2, # Controls number of messages
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

    # Model object with best parameters (** to unpack parameter dict)
    fitted_model = xgb.XGBRegressor(**clf.best_params_)
    
    cvl = cross_val_score(fitted_model, X, y, cv=idc_list, scoring='neg_mean_squared_error')

    print('Cross validation scores per fold: ', cvl)
    print('Mean cross validation score: ', cvl.mean())
    print('Standard deviation: ', cvl.std())

    plot_prediction_per_fold(X, y, fitted_model, idc_list)

    return clf, fitted_model, cvl

# Train model function without plotting
def train_xgb_model_no_plot(X, y, idc_list, params, n_jobs=4, scorer='neg_mean_squared_error', return_train=True):

    # Define model object.
    xgb_model = xgb.XGBRegressor()
    
    # Set up grid search. 
    clf = GridSearchCV(xgb_model, 
                       params, 
                       cv=idc_list, # Int or iterator (default for int is kfold)
                       verbose=2, # Controls number of messages
                       n_jobs=n_jobs, # No of parallell jobs
                       scoring=scorer, # Can use multiple metrics
                       refit=True, # Default True. For multiple metric evaluation, refit must be str denoting scorer to be used 
                       #to find the best parameters for refitting the estimator.
                       return_train_score=return_train) # Default False. If False, cv_results_ will not include training scores.

    # Fit model to folds
    clf.fit(X, y)

    # Model object with best parameters (** to unpack parameter dict)
    fitted_model = xgb.XGBRegressor(**clf.best_params_)

    # Get cross validation scores
    cvl = cross_val_score(fitted_model, X, y, cv=idc_list, scoring='neg_mean_squared_error')

    return clf, fitted_model, cvl

# Functions for data processing
# Reshape dataframe to monthly resolution
def reshape_dataset_monthly(df, id_vars, variables, months_order): 

    df_list = []

    for var in variables:
        # Filter columns for the current variable and the ID columns
        cols = [col for col in df.columns if col.startswith(var) or col in id_vars]
        df_var = df[cols]

        # Rename the columns to have just the month
        df_var = df_var.rename(columns=lambda x: x.split('_')[-1] if x not in id_vars else x)

        # Melt the DataFrame to long format and add month order
        df_melted = pd.melt(df_var, id_vars=id_vars, var_name='month', value_name=var)
        df_melted['month'] = pd.Categorical(df_melted['month'], categories=months_order, ordered=True)

        df_list.append(df_melted)

    # Combine all reshaped DataFrames
    df_final = df_list[0]
    for df_temp in df_list[1:]:
        df_final = pd.merge(df_final, df_temp, on=id_vars + ['month'], how='left')

    # Sort the DataFrame based on ID variables and month
    df_final = df_final.sort_values(by=id_vars + ['month'])

    return(df_final)

# Custom objective function scikit learn api with metadata, to be used with custom XGBRegressor class
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
    
    # Initialize empty arrays for gradient and hessian
    gradients = np.zeros_like(y_pred)
    hessians = np.ones_like(y_pred) # Ones in case of mse
    
    # Unique aggregation groups based on the aggregation ID
    unique_ids = np.unique(metadata[:, 0])
    
    # Loop over each unique ID to aggregate accordingly
    for uid in unique_ids:
        # Find indexes for the current aggregation group
        indexes = metadata[:, 0] == uid
        
        # Aggregate y_pred for the current group
        y_pred_agg = np.sum(y_pred[indexes])
        
        # True value is the same repeated value for the group, so we can use the mean
        y_true_mean = np.mean(y_true[indexes])
        
        # Compute gradients for the group based on the aggregated prediction
        gradient = y_pred_agg - y_true_mean
        gradients[indexes] = gradient

    return gradients, hessians
    