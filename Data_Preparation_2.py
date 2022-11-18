# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:48:01 2022

@author: Cory
"""
import pandas as pd
import numpy as np
from Data_Preparation_Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None)
train_df = pd.read_csv('raw_data_modified.csv', low_memory=False)

# create a list of features that vary, this will be used to create one vector
# for each play. These will be player specific features
player_features = []
for col in train_df.columns:
    if col == 'Position_FB':
        player_features.append('Position_FB')
    if train_df[col][:22].var()!= 0:
        player_features.append(col)

X = np.array(train_df[player_features]).reshape(-1, len(player_features)*22)

# create a dataframe that contains all the features that do not differ per player in play
const_features_play_df = train_df.drop(columns=player_features)
const_features_cols = const_features_play_df.columns

a = const_features_play_df.shape[1] # number of features that are the same for each player, for each play
b = train_df.shape[0] # number of rows
d = X.shape[1] # number of features X currently has 
n = X.shape[0]
m = np.zeros((n,a-1)) # remove a column for the Y values
# add filler matrix m to X
X = np.concatenate((X, m), axis=1)

# loop through each play and add the constant features for each play
# to the overall feature vector X (these features will take place of the 0's inserted)
# also create Y vector (predictor)

Y = []
for i in range(0,b,22):
    Y.append(const_features_play_df.iloc[i].Yards)
    features = np.array(const_features_play_df.iloc[i].drop(labels='Yards'))
    row_index = int(i/22)
    X[row_index,d:] = features

Y = np.array(Y)

# standardize columns of X that are not one hot encoded or categorical-like features
cols_feat = [0,1,2,3,4,6,7,8,9,10,11]
X_cols = []
# for each player (22 total) involved in the play
for i in range(22):
    c = i*21
    for val in cols_feat:
        X_cols.append(val+c)
        
X_cols = np.array(X_cols)    
cols_const = np.array([463,464,465,467,469,470,471,472,473,474])

all_features = np.concatenate((X_cols, cols_const))

# create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=42)

# create standard scalar instance to standardize data
std_scaler = StandardScaler() 
X_train[:,all_features] = std_scaler.fit_transform(X_train[:,all_features])
X_val[:,all_features] = std_scaler.transform(X_val[:,all_features])

# X_train = pd.DataFrame(X_train)
# X_train.to_csv('X_train.csv', index=False)
# X_val = pd.DataFrame(X_val)
# X_val.to_csv('X_val.csv', index=False)
# y_train = pd.DataFrame(y_train)
# y_train.to_csv('y_train.csv', index=False)
# y_val = pd.DataFrame(y_val)
# y_val.to_csv('y_val.csv', index=False)