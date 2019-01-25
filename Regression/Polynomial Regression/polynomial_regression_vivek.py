# Polynomial Regression


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


# Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)

# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
polynomial_regressor.fit(X_poly, Y)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, Y)


# Visualize Linear Regression
plt.scatter(X, Y, color='red') # original Value
plt.plot(X, linear_regressor.predict(X), color = 'blue') # predection
plt.title('Salary Graph with Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualize Polynomial Regression
X_grid = np.arange(min([X]), max([X]), 0.1) # for Better Prediction
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')  # original Value
plt.plot(X_grid, linear_regressor_2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue') # can use polynomial_regressor.fit_transform(X)
plt.title('Salary Graph with Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

linear_regressor.predict(6.5)
linear_regressor_2.predict(polynomial_regressor.fit_transform(6.5))

linear_regressor.predict(8.5)
linear_regressor_2.predict(polynomial_regressor.fit_transform(8.5))


# to Export Model to File
'''from sklearn.externals import joblib
filename = 'Linear_Regression_Model_For_Salary.sav'
joblib.dump(linear_regressor, filename) '''

