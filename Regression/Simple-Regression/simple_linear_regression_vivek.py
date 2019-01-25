# Simple Linear Regression
# Data Preprocessing

import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset
from sklearn.cross_validation import train_test_split

# Import dataset
dataset  = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predecting the Test set Result
# Predection Vector
Y_Pred = regressor.predict(X_test)
my_pred = regressor.predict(30.5)

# Visualize Training set Results

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualize Test set Results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()