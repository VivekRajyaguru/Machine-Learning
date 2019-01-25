# Multiple Linear Regression

# Data Preprocessing

import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset
from sklearn.cross_validation import train_test_split

# Import dataset
dataset  = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Label Encoder
# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder_X = OneHotEncoder(categorical_features = [3])
X = onehotencoder_X.fit_transform(X).toarray()


# Avoding Dummy Variable trap
X = X[:, 1:]

# Splitting data 
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Fit Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting Test Result
# Predection Vector
Y_Pred = regressor.predict(X_test)

# Build optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


pvalues = []
# Method for Backward Elimination with p-value
def backwardElimination(x, sl):
    columns = len(x[0])
    #print(columns) # no of columns
    for i in range(0, columns):
        regressor_OLS = sm.OLS(endog = Y, exog = x).fit()
        pvalues.append(regressor_OLS.pvalues.astype(float))        
        print(pvalues)
        maxValue = max(regressor_OLS.pvalues).astype(float)
        if maxValue > sl:
            for j in range(0, columns - i):
                if regressor_OLS.pvalues[j].astype(float) == maxValue:
                    print(j)
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
    
X_modeled = backwardElimination(X_opt, 0.05)
