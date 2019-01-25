# Decision Tree Regression

# Data Preprocessing
import numpy as np # numeric operation
import matplotlib.pyplot as plt # to plot charts
import pandas as pd # to manage dataset

# Import dataset
dataset  = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # x is always matrix
y = dataset.iloc[:, 2].values # y is always vector


# feature scalling
'''from sklearn.preprocessing import StandardScaler
y = y.reshape(-1,1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)  '''


# Fitting Decision Tree Regressor to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# prediction
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree results for higher resolution
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()