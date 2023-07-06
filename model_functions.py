# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib.pyplot as plt
import seaborn as sns

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