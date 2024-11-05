# Author: Kamilla Hauknes Sjursen (kasj@hvl.no) November 2024

# Import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns


#%%%%%%%%%%% PLOTS CROSS-VALIDATION %%%%%%%%%%%%

# Bar plot of number of annual and seasonal per fold
def plot_train_val_counts(df_train_X, splits_s):
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    # Create a color map or list for the bars
    colors = ['C0', 'C1']

    n_months_to_season = {5: 'summer', 7: 'winter', 12: 'annual'}

    for i, (train_index, val_index) in enumerate(splits_s):
        ax = axes[i]

        n_months_train = df_train_X.iloc[train_index]['n_months']
        n_months_val = df_train_X.iloc[val_index]['n_months']
    
        # Counts
        n_months_train_counts = dict(zip(*np.unique(n_months_train, return_counts=True)))
        n_months_val_counts = dict(zip(*np.unique(n_months_val, return_counts=True)))
    
        n_months_values = sorted(set(n_months_train_counts.keys()).union(n_months_val_counts.keys()))
        season_names = [n_months_to_season[n_months] for n_months in n_months_values]

        train_positions = np.arange(len(n_months_values))
        val_positions = train_positions + 0.4 
    
        train_counts = [n_months_train_counts.get(x, 0) for x in n_months_values]
        ax.bar(train_positions, train_counts, width=0.4, label='Train', color=colors[0])
    
        val_counts = [n_months_val_counts.get(x, 0) for x in n_months_values]
        ax.bar(val_positions, val_counts, width=0.4, label='Validation', color=colors[1])

        # Annotate each bar with the respective count
        for j in range(len(n_months_values)):
            train_count = n_months_train_counts.get(n_months_values[j], 0)
            val_count = n_months_val_counts.get(n_months_values[j], 0)
            train_pos = train_positions[j]
            val_pos = val_positions[j]
        
            ax.text(train_pos, train_count + max(train_count, val_count) * 0.01, str(train_count),
                    ha='center', va='bottom', fontsize=8, color='k')

            ax.text(val_pos, val_count + max(train_count, val_count) * 0.01, str(val_count),
                    ha='center', va='bottom', fontsize=8, color='k')
    
        ax.set_title(f'Fold {i+1}')
        ax.set_ylabel('Count')
        ax.set_xticks(train_positions + 0.2) 
        ax.set_xticklabels(season_names)
    
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()

    
# Plot feature distributions
def plot_feature_distributions(df_X, feature_names, type='train', folds=5, splits_s=None):
    
    fig, axes = plt.subplots(len(feature_names), folds, figsize=(4*folds, len(feature_names) * folds))

    if type=='train':
        for row, feature in enumerate(feature_names):
            for col, (train_index, val_index) in enumerate(splits_s):

                feature_train = df_X.iloc[train_index][feature]
                feature_val = df_X.iloc[val_index][feature]
        
                feature_data_combined = [feature_train, feature_val]
        
                axes[row, col].boxplot(feature_data_combined, positions=[1, 2], widths=0.6)
        
                if row == 0:
                    axes[row, col].set_title(f'Fold {col+1}')
                if col == 0:
                    axes[row, col].set_ylabel(feature)
        
                axes[row, col].set_xticks([1, 2])
                axes[row, col].set_xticklabels(['Train', 'Val'])
                
    elif type=='test':
        for row, feature in enumerate(feature_names):
            
            feature_test = df_X[feature]  
            axes[row].boxplot(feature_test, positions=[1], widths=0.6)  

            axes[row].set_ylabel(feature)  

            axes[row].set_xticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Plot grid search results
def plot_gsearch_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    width = len(grid.best_params_.keys())*5

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(width,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())
        ax[i].grid()

    plt.legend()
    plt.show()

    
# Plot grid search results setting learning rate threshold for too high learning rates
def plot_gsearch_results_mod(grid, learning_rate_threshold=0.3):
    """
    Params: 
        grid: A trained GridSearchCV object.
        learning_rate_threshold: The maximum learning rate to include in the plot.
    """
    # Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']
    
    params_list = results['params']

    # Create a mask to filter out parameter combinations with learning rates > threshold
    learning_rate_mask = [param['learning_rate'] <= learning_rate_threshold for param in params_list]
    
    filtered_means_test = means_test[learning_rate_mask]
    filtered_stds_test = stds_test[learning_rate_mask]
    filtered_means_train = means_train[learning_rate_mask]
    filtered_stds_train = stds_train[learning_rate_mask]
    filtered_params_list = [params_list[i] for i in range(len(params_list)) if learning_rate_mask[i]]

    # Extract the unique values for each parameter to plot
    unique_param_values = {param_name: sorted(set(param_dict[param_name] for param_dict in filtered_params_list))
                           for param_name in grid.param_grid.keys()}

    width = len(grid.param_grid.keys()) * 5

    # Plot results
    fig, ax = plt.subplots(1, len(unique_param_values), sharex='none', sharey='all', figsize=(width, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

    for i, (param_name, param_values) in enumerate(unique_param_values.items()):
        # For each parameter, extract the means and stds of the scores
        param_results_test = {val: [] for val in param_values}
        param_results_train = {val: [] for val in param_values}
        
        for j, params in enumerate(filtered_params_list):
            param_results_test[params[param_name]].append((filtered_means_test[j], filtered_stds_test[j]))
            param_results_train[params[param_name]].append((filtered_means_train[j], filtered_stds_train[j]))

        # Aggregate the results for plotting
        x = param_values
        y_test = [np.mean([score[0] for score in param_results_test[val]]) for val in x]
        e_test = [np.mean([score[1] for score in param_results_test[val]]) for val in x]
        y_train = [np.mean([score[0] for score in param_results_train[val]]) for val in x]
        e_train = [np.mean([score[1] for score in param_results_train[val]]) for val in x]

        ax[i].errorbar(x, y_test, e_test, linestyle='--', marker='o', label='validation')
        ax[i].errorbar(x, y_train, e_train, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(param_name.upper())
        ax[i].grid()

    plt.legend()
    plt.show()

    
#%%%%%%%% PLOTS PREDICTIONS %%%%%%%%%%%%%%%%


def plot_prediction(y1, y2, data_type:str, n_toplot=10**10, fold=False):
    """
    Plot model predictions y1 vs. actual observations y2 and show
    calculated error metrics.

    Parameters:
    y1 : np.array
        Predicted labels.
    y2 : np.array
        Actual labels.
    data_type : str
        Type of data, e.g. "Validation" or "Test".
    n_toplot : int
        Number of points to plot. 
    """
    
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    if fold:
        figsize=(5,5)
        fontsize=12
        s= 15
    else:
        figsize=(8,8)
        fontsize=16
        s= 20
    
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)

    y_max = 8#7 #max(max(y1), max(y2))[0] + 1
    y_min = -15#1 #min(min(y1), min(y2))[0] - 1
    
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    fig = plt.figure(figsize=figsize)
    plt.title("Model Evaluation " + data_type, fontsize=fontsize)
    plt.ylabel('Modeled SMB (m.w.e)', fontsize=fontsize)
    plt.xlabel('Reference SMB (m.w.e)', fontsize=fontsize)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=s)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc) 
    lineStart = y_min
    lineEnd = y_max
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, squared=False), ),
    r'$MSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, squared=True), ),
    r'$MAE=%.2f$' % (mean_absolute_error(y_expected, y_predicted), ),
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.show()


def plot_prediction_subplot(y1, y2, data_type:str, ax, n_toplot=10**10, fold=False):
    """
    Plot model predictions y1 vs. actual observations y2 and show
    calculated error metrics.

    Parameters:
    y1 : np.array
        Predicted labels.
    y2 : np.array
        Actual labels.
    data_type : str
        Type of data, e.g. "Validation" or "Test".
    ax : array
        Axis object
    n_toplot : int
        Number of points to plot. 
    """
    
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

    if fold:
        figsize=(5,5)
        fontsize=12
        s= 15
    else:
        figsize=(8,8)
        fontsize=16
        s= 20
    
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)

    y_max = 8#7 #max(max(y1), max(y2))[0] + 1
    y_min = -15#1 #min(min(y1), min(y2))[0] - 1
    
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    sc = ax.scatter(y_plt, ann_plt, c=z, s=s)
    sc.set_clim(0,0.2)
    plt.colorbar(sc,ax=ax,fraction=0.046) 
    lineStart = y_min
    lineEnd = y_max
    ax.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    ax.set_title("Model Evaluation " + data_type, fontsize=fontsize)
    ax.set_ylabel('Modeled SMB (m.w.e)', fontsize=fontsize)
    ax.set_xlabel('Reference SMB (m.w.e)', fontsize=fontsize)
    ax.axvline(0.0, ls='-.', c='k')
    ax.axhline(0.0, ls='-.', c='k')
    ax.set_xlim(lineStart, lineEnd)
    ax.set_ylim(lineStart, lineEnd)
    ax.set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.2f$' % (root_mean_squared_error(y_expected, y_predicted), ),
    r'$MSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, ), ),
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    return ax


def plot_prediction_per_season(y_test_all, y_pred_all, season='Annual'):

    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
    
    figsize=(5,5)
    fontsize=16
    s= 20
    n_toplot=10**10
    
    idxs = np.arange(len(y_test_all))
    np.random.shuffle(idxs)

    y_max = 8#7 #max(max(y1), max(y2))[0] + 1
    y_min = -15#1 #min(min(y1), min(y2))[0] - 1
    
    y_expected = y_test_all.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y_pred_all.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    fig = plt.figure(figsize=figsize, dpi=100)
    plt.title(season + " mass balance", fontsize=20)
    plt.ylabel('Modeled mass balance (m w.e)', fontsize=fontsize)
    plt.xlabel('Observed mass balance (m w.e)', fontsize=fontsize)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=s)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    #plt.colorbar(sc) 
    lineStart = y_min
    lineEnd = y_max
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.2f$' % (root_mean_squared_error(y_expected, y_predicted), ),
    r'$MSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, ), ),
    r'$MAE=%.2f$' % (mean_absolute_error(y_expected, y_predicted), ),
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


#%%%%%%%%% FEATURE IMPORTANCE %%%%%%%%%%%%%%


# Plot feature importance for a trained model
def plot_feature_importance(best_model):
    #best_model = clf_loaded.best_estimator_
    
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(df_train_X.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")
    plt.show()


# Plot permutation importance
def plot_permutation_importance_per_fold(df_train_X_s, X_train_s, y_train_s, splits_s, best_model, max_features_plot = 10):

    fig, ax = plt.subplots(1,5, figsize=(30,10))
    a = 0    
    for train_index, test_index in splits_s:
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X_train_s[train_index], X_train_s[test_index]
        y_train, y_test = y_train_s[train_index], y_train_s[test_index]

        best_model.fit(X_train, y_train)
    
        result = permutation_importance(best_model, X_train, y_train, n_repeats=20, random_state=42, n_jobs=10)

        result.importances_mean[-3:]=0.0
        sorted_idx = result.importances_mean.argsort()
        labels = np.array(df_train_X_s.columns)[sorted_idx][-max_features_plot:]
    
        ax[a].boxplot(result.importances[sorted_idx].T[:,-max_features_plot:], vert=False, labels=labels)
        ax[a].set_title("Permutation Importance Fold " + str(a))

        a=a+1

    fig.show()

    return result


''' OLD

def plot_prediction_subplot(y1, y2, data_type:str, ax, n_toplot=10**10, fold=False):
    """
    Plot model predictions y1 vs. actual observations y2 and show
    calculated error metrics.

    Parameters:
    y1 : np.array
        Predicted labels.
    y2 : np.array
        Actual labels.
    data_type : str
        Type of data, e.g. "Validation" or "Test".
    ax : array
        Axis object
    n_toplot : int
        Number of points to plot. 
    """
    
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error

    if fold:
        figsize=(5,5)
        fontsize=12
        s= 15
    else:
        figsize=(8,8)
        fontsize=16
        s= 20
    
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)

    y_max = 8#7 #max(max(y1), max(y2))[0] + 1
    y_min = -15#1 #min(min(y1), min(y2))[0] - 1
    
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    #fig = plt.figure(figsize=figsize)
    ax.scatter(y_plt, ann_plt, c=z, s=s)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc) 
    lineStart = y_min
    lineEnd = y_max
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.title("Model Evaluation " + data_type, fontsize=fontsize)
    plt.ylabel('Modeled SMB (m.w.e)', fontsize=fontsize)
    plt.xlabel('Reference SMB (m.w.e)', fontsize=fontsize)
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, squared=False), ),
    r'$MSE=%.2f$' % (mean_squared_error(y_expected, y_predicted, squared=True), ),
    r'$MAE=%.2f$' % (mean_absolute_error(y_expected, y_predicted), ),
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    return ax
    #plt.show()

def plot_prediction_per_fold(X, y, model, idc_list):
    """
    Plot model predictions of model vs. test data y based on 
    folds given by indices X_idc and y_idc.
  
    Parameters:
    X : np.array
        Training dataset of features.
    y : np.array
        Labels of training dataset.
    model : sklearn.model
        Fitted XGBmodel object.
    X_idc : np.array
        Indices of fold splits of features.
    y_idc : np.array
        Indices of fold splits for labels.
    """

    y_test_list = []
    y_pred_list = []
    i = 0

    for train_index, test_index in idc_list:
        # Loops over n_splits iterations and gets train and test splits in each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_list.extend(y_test)
        y_pred_list.extend(y_pred)

        title = 'Validation fold ' + str(i)

        plot_prediction(y_test, y_pred, title, n_toplot=5000, fold=True)

        i=i+1

    # Arrays of predictions and observations for each fold
    y_test_all = np.hstack([*y_test_list])
    y_pred_all = np.hstack([*y_pred_list])

    # Plot predictions and observations for each cross-valiadation fold together
    plot_prediction(y_test_all, y_pred_all, 'Validation', n_toplot=12000)

def plot_feature_permutation_importance(model, X, y, X_labels):

    model.fit(X, y)
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X_labels[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=X_labels[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()

'''