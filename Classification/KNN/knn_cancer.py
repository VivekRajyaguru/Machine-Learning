# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BreastCancer.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values

# For Missing Data
# for Data preprocessing Imputer will allow to handle Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Build optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_opt = X[:, [0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Fitting the Classifier to the dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,p=2, metric='minkowski')
classifier.fit(X_train, y_train)


# Fitting using Logistic Regression model
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression(random_state = 0)
regression.fit(X_train,y_train)


# Predicting a new result
y_pred = classifier.predict(X_test)
my_pred = classifier.predict(sc_X.transform([[3,2,1,2,2,1,3,1,1]]))

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Export Model
from sklearn.externals import joblib
fileName = 'Breast-cancer-model.sav'
joblib.dump(classifier, fileName)


# Import Model and Predict
'''from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
fileName = 'Breast-cancer-model.sav'
classification_model = joblib.load(fileName)
sc_2 = StandardScaler()
prediction = classification_model.predict([[1,1,1,1,1,1,1,1,1]])'''





