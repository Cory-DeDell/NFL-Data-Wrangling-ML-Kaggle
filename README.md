# NFL-Data-Wrangling-ML-Kaggle
This repository contains code to wrangle the data in the NFL Big Data Bowl 2020 dataset on Kaggle. Also included is code to perform a machine learning
task to predict yards gained by an NFL rusher based on the data set features.

# Data Preparation
Data_Preparation_1.py contains code to correct issues with the raw kaggle data as well as construct feature engineering on the data. Also removes unnecessary features. 

Data_Preparation_Functions.py cointains fucntions used in Data_Preparation_1.py

Data_Preparation_2.py contains code to construct a matrix of the prepared data for every play and create a testing and training set that can be used to perform machine learning tasks.

# ML utility functions
Utils.py contains functions used for training and testing the machine learning model.

# Model
Model.py contains the code to train and evaluate the specified machine learning model.
