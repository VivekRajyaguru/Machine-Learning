# Multiple Linear Regression

# Data Preprocessing

import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset
from sklearn.cross_validation import train_test_split

# Import dataset
dataset  = pd.read_csv("College.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 17].values

# Label Encoder
# Encode categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# Splitting data 
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fit Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_modeled, Y_train)

# Predicting Test Result
# Predection Vector
Y_Pred = regressor.predict(X_test_modeled)

# Build optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((518,1)).astype(int), values = X_train , axis = 1)
X_test = np.append(arr = np.ones((259,1)).astype(int), values = X_test , axis = 1)
X_opt = X_train[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_opt_test = X_test[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

# Method for Backward Elimination with p-value
pvalues = []
def backwardElimination(x, sl):
    columns = len(x[0])
    for i in range(0, columns):
        regressor_OLS = sm.OLS(endog = Y_test, exog = x).fit()
        maxValue = max(regressor_OLS.pvalues).astype(float)
        if maxValue > sl:
            for j in range(0, columns - i):
                if regressor_OLS.pvalues[j].astype(float) == maxValue:
                    print(j)
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
    
X_modeled = backwardElimination(X_opt, 0.05)
X_test_modeled = backwardElimination(X_opt_test, 0.05)
