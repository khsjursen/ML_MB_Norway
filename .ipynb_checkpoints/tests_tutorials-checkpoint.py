# -*- coding: utf-8 -*-
"""
KHS 03.03.2023

Tests, tutorials XGBoost

"""

#%% Iris classification

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# iris dataset contains 150 instances of four features of flowers 
# (sepal length, sepal width, petal length, petal width). 
# Variable target has values 0-2 for each of the species setosa, versicolor, virginica
# Data contains 50 instances for each target. 
data = load_iris()

# Split dataset into training and testing sets. 'data' is x values, 'target' is y. 
# Split with 80% train and 20% test.
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.4)

# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

# fit model
bst.fit(X_train, y_train)

# make predictions
preds = bst.predict(X_test)

#%% Diamonds regression from datacamp.com

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# Load diamonds dataset
diamonds = sns.load_dataset("diamonds")

# extracting the feature and target arrays based on the dataset
# One feature dataset (X) with all features except price and one target 
# dataset (y) with only price
X, y = diamonds.drop('price', axis=1), diamonds[['price']]

# The dataset has three categorical features.
#  XGBoost has the ability to internally deal with categoricals
# Cast categorical columns into Pandas category data type

# Extract text features
# cats are ['cut','color','clarity']
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')

# X now contains categorical variables for  ['cut','color','clarity']
# Use X.dtypes() to check categories:
# carat       float64
# cut        category
# color      category
# clarity    category
# depth       float64
# table       float64
# x           float64
# y           float64
# z           float64

# Split data into train and test sets.
# 0.25 test size is default
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# XGboost class for storing datasets is called DMatrix
# Create regression matrices:
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# After building the DMatrices, you should choose a value for the objective parameter. 
# It tells XGBoost the machine learning problem you are trying to solve and what 
# metrics or loss functions to use to solve that problem.
#For example, to predict diamond prices, which is a regression problem, you can 
# use the common reg:squarederror objective. 
# mse = np.mean((actual - predicted) ** 2)
# rmse = np.sqrt(mse)

# A note on the difference between a loss function and a performance metric: 
# A loss function is used by machine learning models to minimize the differences 
# between the actual (ground truth) values and model predictions. On the other hand, 
# a metric (or metrics) is chosen by the machine learning engineer to measure the 
# similarity between ground truth and model predictions.
# In short, a loss function should be minimized while a metric should be maximized. 
# A loss function is used during training to guide the model on where to improve. 
# A metric is used during evaluation to measure overall performance.

# Training:
# Define hyperparameters and objective function by dicitionary. 
params = {"objective": "reg:squarederror", "tree_method": "hist"}

# Set number of boosting rounds. XGBoost minimizes the loss function RMSE in small 
# incremental rounds, num_boost_rounds sets amount of rounds. Ideal number of rounds
# is found through hyperparameter tuning. For now, set to 100.
n = 100

# Train model:
model = xgb.train(
    params = params,
    dtrain = dtrain_reg,
    num_boost_round = n,
    )

# Measure performance of model by testing on unseen data:
    
# Generate predictions based on model
preds = model.predict(dtest_reg)

# compare predtictions against test data y_test
rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")

#%% Using validation sets during training

# Use evaluation arrays that allow us to see model performance as it gets 
# improved incrementally across boosting rounds.

# Set up parameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}
n = 100

# Create a list of two tuples that each contain two elements. The first element 
# is the array for the model to evaluate, and the second is the array’s name.
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

# When we pass this array to the evals parameter of xgb.train, we will see the 
# model performance after each boosting round:
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval = 10 # Prints performance updates every 10 boosting rounds
   )

#%% XGboost early stopping

# Set up parameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}
n = 5000

# Create a list of two tuples that each contain two elements. The first element 
# is the array for the model to evaluate, and the second is the array’s name.
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

# When we pass this array to the evals parameter of xgb.train, we will see the 
# model performance after each boosting round:
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval = 250 # Prints performance updates every 250 boosting rounds
   )

# When given an uneccesary number of boosting rounds, overfitting occurs 
# (training loss goes down, but validation loss increases)
# Early stopping: technique that forces XGBoost to watch validation loss and
# if it stops improving for a specified number of rounds, it automatically stops
# training. 

# Parameter early_stopping_rounds indicates that training should be stopped
# if validation loss doesn't improve for 50 consecutive rounds. 
# NOTE: If there is more than one item in evals, the LAST entry will be used for
# early stopping!
# If there’s more than one metric in the eval_metric parameter given in params, 
# the last metric will be used for early stopping.

n = 10000

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=50,
   # Activate early stopping
   early_stopping_rounds=50
   )

#%% Cross-validation

# Using cross-validation to avoid splitting dataset into three. In cross-validation
# we split the training set into k folds. Then we train the model k times. Each time,
# we use k-1 parts for training and the final kth part for validation. Called
# k-fold cross-validation. After all folds are done, we can take the mean of the
# scores as the final, most realistic performance of the model. 

# Use cv function of XGBoost for cross-validation
params = {"objective": "reg:squarederror", "tree_method": "hist"}

n = 1000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n,
   nfold=5,
   early_stopping_rounds=20
   )

# Get best score by taking minimum of test-rmse-mean column:
best_rmse = results['test-rmse-mean'].min()

# Note that this method of cross-validation is used to see the true performance 
# of the model. Once satisfied with its score, you must retrain it on the full 
# data before deployment.



