# Multiple Linear Regression

# Data Preprocessing

import numpy as np # numeric operation
import pandas as pd # to manage dataset
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset  = pd.read_csv("patient_data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values


# Splitting data 
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fit Multiple Linear Regression to Training Set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting Test Result
# Predection Vector
Y_Pred = regressor.predict(X_test)
test_data = np.zeros((1,2))
test_data[0][0] = 90
test_data[0][1] = 80
my_Pred = regressor.predict(test_data)