# Multiple Linear Regression

# Data Preprocessing

import numpy as np # numeric operation
import pandas as pd # to manage dataset
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Import dataset
dataset  = pd.read_csv("fuel-consumption.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 4]

# Spliting dataset to Training and Test Set
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scalling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
