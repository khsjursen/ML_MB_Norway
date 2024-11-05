# Author: Kamilla Hauknes Sjursen (kasj@hvl.no) November 2024

import numpy as np
import pandas as pd

# Using GroupKFold to split on id
'''
# Get glacier IDs from training dataset (in the order of which they appear in training dataset).
# gp_s is an array with shape equal to the shape of X_train_s and y_train_s.
gp_s = np.array(df_train_final['id'].values)

# Use five folds
group_kf = GroupKFold(n_splits=5)

# Split into folds according to group by glacier ID.
# For each unique glacier ID, indices in gp_s indicate which rows in X_train_s and y_train_s belong to the glacier.
splits_s = list(group_kf.split(X_train, y_train, gp_s))

#print('Train, fold 0: ', np.unique(gp_s[splits_s[0][0]]))
#print('Validation, fold 0: ', np.unique(gp_s[splits_s[0][1]]))
#print('Train, fold 1: ', np.unique(gp_s[splits_s[1][0]]))
#print('Validation, fold 1: ', np.unique(gp_s[splits_s[1][1]]))
#print('Train, fold 2: ', np.unique(gp_s[splits_s[2][0]]))
#print('Validation, fold 2: ', np.unique(gp_s[splits_s[2][1]]))
#print('Train, fold 3: ', np.unique(gp_s[splits_s[3][0]]))
#print('Validation, fold 3: ', np.unique(gp_s[splits_s[3][1]]))
#print('Train, fold 4: ', np.unique(gp_s[splits_s[4][0]]))
#print('Validation, fold 4: ', np.unique(gp_s[splits_s[4][1]]))
print(len(gp_s))
print(y_train.shape)
print(X_train.shape)
print(df_train_X.columns)
print(df_train_y.columns)

# Check fold indices for training/validation data
fold_indices = []

for train_index, val_index in group_kf.split(X_train, y_train, gp_s):
    print("TRAIN:", train_index, "VALIDATION:", val_index)
    print("shape(train):", train_index.shape, "test:", val_index.shape)
    fold_indices.append((train_index, val_index))
    '''

# USING CUSTOM ITERATOR TO SPLIT ON YEAR INTERVALS
'''
# Define the year intervals for the folds
year_intervals = [
    (1960, 1969),  # Fold 1
    (1970, 1979),  # Fold 2
    (1980, 1994),  # Fold 3
    (1995, 2009),  # Fold 4
    (2010, 2021)   # Fold 5
]

# Add a 'fold' column based on the 'year' intervals
def assign_fold(row):
    for i, (start_year, end_year) in enumerate(year_intervals):
        if start_year <= row['year'] <= end_year:
            return i
    return -1  # Return -1 if year is not in any interval (should not happen)

df_train_final['fold'] = df_train_final.apply(assign_fold, axis=1)

# Verify that all rows have been assigned a valid fold
if (df_train_final['fold'] == -1).any():
    raise ValueError("Some rows have not been assigned a valid fold")

# Group by 'id' to maintain groups of rows with the same 'id'
grouped = df_train_final.groupby('id')

# Create indices for each fold
folds = [([], []) for _ in range(5)]  # Initialize with 5 empty train/test index lists

# Distribute groups into folds
for _, group in grouped:
    fold = group['fold'].iloc[0]  # All rows in group have the same fold
    for fold_idx in range(5):
        if fold == fold_idx:
            folds[fold_idx][1].extend(group.index)  # Assign group to test set of this fold
        else:
            folds[fold_idx][0].extend(group.index)  # Assign group to train set of other folds

# Convert lists to numpy arrays
folds = [(np.array(train_indices), np.array(test_indices)) for train_indices, test_indices in folds]


## Splitting the DataFrame into folds for cross-validation
#folds = []
#for fold in range(5):
#    train_indices = df[df['fold'] != fold].index
#    test_indices = df[df['fold'] == fold].index
#    folds.append((train_indices, test_indices))

# Fold iterator function
#def get_fold_iterator(data, folds):
#    for train_indices, test_indices in folds:
#        yield data.iloc[train_indices], data.iloc[test_indices]

# Define the custom cross-validator
class CustomFoldIterator(BaseCrossValidator):
    def __init__(self, fold_indices):
        self.fold_indices = fold_indices

    def split(self, X, y=None, groups=None):
        for train_indices, test_indices in self.fold_indices:
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_indices)

# Setup and Execute GridSearchCV
custom_cv = CustomFoldIterator(folds)

# Create splits
splits_s = list(custom_cv.split(X_train, y_train))

# Convert Int64Index to numpy arrays 
splits_s = [(np.array(train_indices), np.array(test_indices)) for (train_indices, test_indices) in splits_s]

# Print number of instances in each split
for i, (train_indices, test_indices) in enumerate(splits_s):
    print(f"Fold {i+1} - Train: {len(train_indices)}, Test: {len(test_indices)}")

# Check fold indices for training/validation data
fold_indices = []

for i, (train_index, val_index) in enumerate(splits_s):
    print(f"Fold {i+1}")
    print("TRAIN:", train_index)
    print("VALIDATION:", val_index)
    print("shape(train):", train_index.shape, "shape(validation):", val_index.shape)
    fold_indices.append((train_index, val_index))
    '''