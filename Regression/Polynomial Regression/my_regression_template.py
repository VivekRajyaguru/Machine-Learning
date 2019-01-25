# Regression Template

# Data Preprocessing
import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset

# Import dataset
dataset  = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # x is always matrix
Y = dataset.iloc[:, 2].values # y is always vector

# Splitting data 
# Due to lack of data no need for splitting
'''from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)'''


# Fitting Regression Model to dataset

# Visualize Polynomial Regression
