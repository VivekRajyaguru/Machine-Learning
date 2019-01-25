# Data Preprocessing

import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset

# Import dataset

dataset  = pd.read_csv("Data.csv")
# iloc get data from dataset
# [rows index, columns index]
# : = all row/colum
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# For Missing Data
# for Data preprocessing Imputer will allow to handle Missing Data
from sklearn.preprocessing import Imputer

# missing_values = missing value to replace
# strategy = imputation strategy
#           mean = replace missing values using mean along with axis
#           median = replace missing values using median along with axis
#           most_frequent = replace missing using most frequent value along with axis
# axis = default 0
#        0 = impute with columns
#        1 = impute with rows
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder_X = OneHotEncoder(categorical_features = [0])
X = onehotencoder_X.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Spliting dataset to Training and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




