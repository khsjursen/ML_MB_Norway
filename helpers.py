# Author: Kamilla Hauknes Sjursen (kasj@hvl.no) November 2024

# Import libraries
import numpy as np
import pandas as pd

#%%%%% PRE-TRAINING DATA PROCESSING %%%%%

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



#%%%%% CROSS-VALIDATION %%%%%


#%%%%% PREDICTION PROCESSING %%%%%%%
    
# Get mass balance predictions for a given season (winter, summer annual)
def get_prediction_per_season(X_train_s, y_train_s, splits_s, best_model, months=12):
    y_pred_list = []
    y_test_list = []
    i=0

    for train_index, test_index in splits_s:
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X_train_s[train_index], X_train_s[test_index]
        y_train, y_test = y_train_s[train_index], y_train_s[test_index]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        indices = np.argwhere((X_test == months))[:,0]
        y_test_crop = y_test[indices]
        y_pred_crop = y_pred[indices]

        y_test_list.extend(y_test_crop)
        y_pred_list.extend(y_pred_crop)

        i=i+1

    # Arrays of predictions and observations for each fold
    y_test_all = np.hstack([*y_test_list])
    y_pred_all = np.hstack([*y_pred_list])

    return y_test_all, y_pred_all

def get_prediction_per_season_test(X_test, y_test, best_model, months=12):

    y_pred = best_model.predict(X_test)

    indices = np.argwhere((X_test[:,-2] == months))[:,0]
    y_test_crop = y_test[indices]
    y_pred_crop = y_pred[indices]

    return y_test_crop, y_pred_crop

